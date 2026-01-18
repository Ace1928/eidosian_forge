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
def rescaleData(data, scale, offset, dtype=None, clip=None):
    """Return data rescaled and optionally cast to a new dtype.

    The scaling operation is::

        data => (data-offset) * scale
    """
    if dtype is None:
        out_dtype = data.dtype
    else:
        out_dtype = np.dtype(dtype)
    if out_dtype.kind in 'ui':
        lim = np.iinfo(out_dtype)
        if clip is None:
            clip = (lim.min, lim.max)
        clip = (max(clip[0], lim.min), min(clip[1], lim.max))
        clip = [math.trunc(x) for x in clip]
    if np.can_cast(data, np.float32):
        work_dtype = np.float32
    else:
        work_dtype = np.float64
    cp = getCupy()
    if cp and cp.get_array_module(data) == cp:
        data_out = data.astype(work_dtype, copy=True)
        data_out -= offset
        data_out *= scale
        if clip is not None:
            clip_array(data_out, clip[0], clip[1], out=data_out)
        return data_out.astype(out_dtype, copy=False)
    numba_fn = getNumbaFunctions()
    if numba_fn and clip is not None:
        return numba_fn.rescaleData(data, scale, offset, out_dtype, clip)
    return _rescaleData_nditer(data, scale, offset, work_dtype, out_dtype, clip)