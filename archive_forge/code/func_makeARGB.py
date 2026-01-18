from __future__ import division
import decimal
import math
import re
import struct
import sys
import warnings
from collections import OrderedDict
import numpy as np
from . import Qt, debug, getConfigOption, reload
from .metaarray import MetaArray
from .Qt import QT_LIB, QtCore, QtGui
from .util.cupy_helper import getCupy
from .util.numba_helper import getNumbaFunctions
def makeARGB(data, lut=None, levels=None, scale=None, useRGBA=False, maskNans=True, output=None):
    """
    Convert an array of values into an ARGB array suitable for building QImages,
    OpenGL textures, etc.
    
    Returns the ARGB array (unsigned byte) and a boolean indicating whether
    there is alpha channel data. This is a two stage process:
    
        1) Rescale the data based on the values in the *levels* argument (min, max).
        2) Determine the final output by passing the rescaled values through a
           lookup table.
   
    Both stages are optional.
    
    ============== ==================================================================================
    **Arguments:**
    data           numpy array of int/float types. If 
    levels         List [min, max]; optionally rescale data before converting through the
                   lookup table. The data is rescaled such that min->0 and max->*scale*::
                   
                      rescaled = (clip(data, min, max) - min) * (*scale* / (max - min))
                   
                   It is also possible to use a 2D (N,2) array of values for levels. In this case,
                   it is assumed that each pair of min,max values in the levels array should be 
                   applied to a different subset of the input data (for example, the input data may 
                   already have RGB values and the levels are used to independently scale each 
                   channel). The use of this feature requires that levels.shape[0] == data.shape[-1].
    scale          The maximum value to which data will be rescaled before being passed through the 
                   lookup table (or returned if there is no lookup table). By default this will
                   be set to the length of the lookup table, or 255 if no lookup table is provided.
    lut            Optional lookup table (array with dtype=ubyte).
                   Values in data will be converted to color by indexing directly from lut.
                   The output data shape will be input.shape + lut.shape[1:].
                   Lookup tables can be built using ColorMap or GradientWidget.
    useRGBA        If True, the data is returned in RGBA order (useful for building OpenGL textures). 
                   The default is False, which returns in ARGB order for use with QImage 
                   (Note that 'ARGB' is a term used by the Qt documentation; the *actual* order 
                   is BGRA).
    maskNans       Enable or disable masking NaNs as transparent.
    ============== ==================================================================================
    """
    cp = getCupy()
    xp = cp.get_array_module(data) if cp else np
    profile = debug.Profiler()
    if data.ndim not in (2, 3):
        raise TypeError('data must be 2D or 3D')
    if data.ndim == 3 and data.shape[2] > 4:
        raise TypeError('data.shape[2] must be <= 4')
    if lut is not None and (not isinstance(lut, xp.ndarray)):
        lut = xp.array(lut)
    if levels is None:
        if data.dtype.kind == 'u':
            levels = xp.array([0, 2 ** (data.itemsize * 8) - 1])
        elif data.dtype.kind == 'i':
            s = 2 ** (data.itemsize * 8 - 1)
            levels = xp.array([-s, s - 1])
        elif data.dtype.kind == 'b':
            levels = xp.array([0, 1])
        else:
            raise Exception('levels argument is required for float input types')
    if not isinstance(levels, xp.ndarray):
        levels = xp.array(levels)
    levels = levels.astype(xp.float64)
    if levels.ndim == 1:
        if levels.shape[0] != 2:
            raise Exception('levels argument must have length 2')
    elif levels.ndim == 2:
        if lut is not None and lut.ndim > 1:
            raise Exception('Cannot make ARGB data when both levels and lut have ndim > 2')
        if levels.shape != (data.shape[-1], 2):
            raise Exception('levels must have shape (data.shape[-1], 2)')
    else:
        raise Exception('levels argument must be 1D or 2D (got shape=%s).' % repr(levels.shape))
    profile('check inputs')
    if scale is None:
        if lut is not None:
            scale = lut.shape[0]
        else:
            scale = 255.0
    if lut is None:
        dtype = xp.ubyte
    else:
        dtype = xp.min_scalar_type(lut.shape[0] - 1)
    nanMask = None
    if maskNans and data.dtype.kind == 'f' and xp.isnan(data.min()):
        nanMask = xp.isnan(data)
        if data.ndim > 2:
            nanMask = xp.any(nanMask, axis=-1)
    if levels is not None:
        if isinstance(levels, xp.ndarray) and levels.ndim == 2:
            if levels.shape[0] != data.shape[-1]:
                raise Exception('When rescaling multi-channel data, there must be the same number of levels as channels (data.shape[-1] == levels.shape[0])')
            newData = xp.empty(data.shape, dtype=int)
            for i in range(data.shape[-1]):
                minVal, maxVal = levels[i]
                if minVal == maxVal:
                    maxVal = xp.nextafter(maxVal, 2 * maxVal)
                rng = maxVal - minVal
                rng = 1 if rng == 0 else rng
                newData[..., i] = rescaleData(data[..., i], scale / rng, minVal, dtype=dtype)
            data = newData
        else:
            minVal, maxVal = levels
            if minVal != 0 or maxVal != scale:
                if minVal == maxVal:
                    maxVal = xp.nextafter(maxVal, 2 * maxVal)
                rng = maxVal - minVal
                rng = 1 if rng == 0 else rng
                data = rescaleData(data, scale / rng, minVal, dtype=dtype)
    profile('apply levels')
    if lut is not None:
        data = applyLookupTable(data, lut)
    elif data.dtype != xp.ubyte:
        data = xp.clip(data, 0, 255).astype(xp.ubyte)
    profile('apply lut')
    if output is None:
        imgData = xp.empty(data.shape[:2] + (4,), dtype=xp.ubyte)
    else:
        imgData = output
    profile('allocate')
    if useRGBA:
        dst_order = [0, 1, 2, 3]
    elif sys.byteorder == 'little':
        dst_order = [2, 1, 0, 3]
    else:
        dst_order = [1, 2, 3, 0]
    fastpath = try_fastpath_argb(xp, data, imgData, useRGBA)
    if fastpath:
        pass
    elif data.ndim == 2:
        for i in range(3):
            imgData[..., dst_order[i]] = data
    elif data.shape[2] == 1:
        for i in range(3):
            imgData[..., dst_order[i]] = data[..., 0]
    else:
        for i in range(0, data.shape[2]):
            imgData[..., dst_order[i]] = data[..., i]
    profile('reorder channels')
    if data.ndim == 3 and data.shape[2] == 4:
        alpha = True
    else:
        alpha = False
        if not fastpath:
            imgData[..., dst_order[3]] = 255
    if nanMask is not None:
        alpha = True
        if xp == cp and tuple(map(int, cp.__version__.split('.'))) < (10, 0):
            imgData[nanMask, :, dst_order[3]] = 0
        else:
            imgData[nanMask, dst_order[3]] = 0
    profile('alpha channel')
    return (imgData, alpha)