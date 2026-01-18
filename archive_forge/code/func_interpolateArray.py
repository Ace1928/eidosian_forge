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
def interpolateArray(data, x, default=0.0, order=1):
    """
    N-dimensional interpolation similar to scipy.ndimage.map_coordinates.
    
    This function returns linearly-interpolated values sampled from a regular
    grid of data. It differs from `ndimage.map_coordinates` by allowing broadcasting
    within the input array.
    
    ==============  ===========================================================================================
    **Arguments:**
    *data*          Array of any shape containing the values to be interpolated.
    *x*             Array with (shape[-1] <= data.ndim) containing the locations within *data* to interpolate.
                    (note: the axes for this argument are transposed relative to the same argument for
                    `ndimage.map_coordinates`).
    *default*       Value to return for locations in *x* that are outside the bounds of *data*.
    *order*         Order of interpolation: 0=nearest, 1=linear.
    ==============  ===========================================================================================
    
    Returns array of shape (x.shape[:-1] + data.shape[x.shape[-1]:])
    
    For example, assume we have the following 2D image data::
    
        >>> data = np.array([[1,   2,   4  ],
                             [10,  20,  40 ],
                             [100, 200, 400]])
        
    To compute a single interpolated point from this data::
        
        >>> x = np.array([(0.5, 0.5)])
        >>> interpolateArray(data, x)
        array([ 8.25])
        
    To compute a 1D list of interpolated locations:: 
        
        >>> x = np.array([(0.5, 0.5),
                          (1.0, 1.0),
                          (1.0, 2.0),
                          (1.5, 0.0)])
        >>> interpolateArray(data, x)
        array([  8.25,  20.  ,  40.  ,  55.  ])
        
    To compute a 2D array of interpolated locations::
    
        >>> x = np.array([[(0.5, 0.5), (1.0, 2.0)],
                          [(1.0, 1.0), (1.5, 0.0)]])
        >>> interpolateArray(data, x)
        array([[  8.25,  40.  ],
               [ 20.  ,  55.  ]])
               
    ..and so on. The *x* argument may have any shape as long as 
    ```x.shape[-1] <= data.ndim```. In the case that 
    ```x.shape[-1] < data.ndim```, then the remaining axes are simply 
    broadcasted as usual. For example, we can interpolate one location
    from an entire row of the data::
    
        >>> x = np.array([[0.5]])
        >>> interpolateArray(data, x)
        array([[  5.5,  11. ,  22. ]])

    This is useful for interpolating from arrays of colors, vertexes, etc.
    """
    if order not in (0, 1):
        raise ValueError('interpolateArray requires order=0 or 1 (got %s)' % order)
    prof = debug.Profiler()
    nd = data.ndim
    md = x.shape[-1]
    if md > nd:
        raise TypeError('x.shape[-1] must be less than or equal to data.ndim')
    totalMask = np.ones(x.shape[:-1], dtype=bool)
    if order == 0:
        xinds = np.round(x).astype(int)
        for ax in range(md):
            mask = (xinds[..., ax] >= 0) & (xinds[..., ax] <= data.shape[ax] - 1)
            xinds[..., ax][~mask] = 0
            totalMask &= mask
        result = data[tuple([xinds[..., i] for i in range(xinds.shape[-1])])]
    elif order == 1:
        fields = np.mgrid[(slice(0, order + 1),) * md]
        xmin = np.floor(x).astype(int)
        xmax = xmin + 1
        indexes = np.concatenate([xmin[np.newaxis, ...], xmax[np.newaxis, ...]])
        fieldInds = []
        for ax in range(md):
            mask = (xmin[..., ax] >= 0) & (x[..., ax] <= data.shape[ax] - 1)
            totalMask &= mask
            mask &= xmax[..., ax] < data.shape[ax]
            axisIndex = indexes[..., ax][fields[ax]]
            axisIndex[axisIndex < 0] = 0
            axisIndex[axisIndex >= data.shape[ax]] = 0
            fieldInds.append(axisIndex)
        prof()
        fieldData = data[tuple(fieldInds)]
        prof()
        s = np.empty((md,) + fieldData.shape, dtype=float)
        dx = x - xmin
        for ax in range(md):
            f1 = fields[ax].reshape(fields[ax].shape + (1,) * (dx.ndim - 1))
            sax = f1 * dx[..., ax] + (1 - f1) * (1 - dx[..., ax])
            sax = sax.reshape(sax.shape + (1,) * (s.ndim - 1 - sax.ndim))
            s[ax] = sax
        s = np.prod(s, axis=0)
        result = fieldData * s
        for i in range(md):
            result = result.sum(axis=0)
    prof()
    if totalMask.ndim > 0:
        result[~totalMask] = default
    elif totalMask is False:
        result[:] = default
    prof()
    return result