import numpy
import cupy
from cupy import _core
from cupy._core import internal
from cupyx.scipy.ndimage import _util
from cupyx.scipy.ndimage import _filters_core
from cupyx.scipy.ndimage import _filters_generic
def derivative2(input, axis, output, mode, cval):
    order = [0] * input.ndim
    order[axis] = 2
    return gaussian_filter(input, sigma, order, output, mode, cval, **kwargs)