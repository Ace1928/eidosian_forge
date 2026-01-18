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
def gaussianFilter(data, sigma):
    """
    Drop-in replacement for scipy.ndimage.gaussian_filter.
    
    (note: results are only approximately equal to the output of
     gaussian_filter)
    """
    cp = getCupy()
    xp = cp.get_array_module(data) if cp else np
    if xp.isscalar(sigma):
        sigma = (sigma,) * data.ndim
    baseline = data.mean()
    filtered = data - baseline
    for ax in range(data.ndim):
        s = sigma[ax]
        if s == 0:
            continue
        ksize = int(s * 6)
        x = xp.arange(-ksize, ksize)
        kernel = xp.exp(-x ** 2 / (2 * s ** 2))
        kshape = [1] * data.ndim
        kshape[ax] = len(kernel)
        kernel = kernel.reshape(kshape)
        shape = data.shape[ax] + ksize
        scale = 1.0 / (abs(s) * (2 * xp.pi) ** 0.5)
        filtered = scale * xp.fft.irfft(xp.fft.rfft(filtered, shape, axis=ax) * xp.fft.rfft(kernel, shape, axis=ax), axis=ax)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(filtered.shape[ax] - data.shape[ax], None, None)
        filtered = filtered[tuple(sl)]
    return filtered + baseline