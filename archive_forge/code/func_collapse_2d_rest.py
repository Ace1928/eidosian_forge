from itertools import product
import cupy
from cupy._core.internal import _normalize_axis_index
from cupy._core._scalar import get_typename
from cupy_backends.cuda.api import runtime
from cupyx.scipy.signal._arraytools import axis_slice
def collapse_2d_rest(x, axis):
    x = cupy.moveaxis(x, axis + 1, -1)
    x_shape = x.shape
    x = x.reshape(x.shape[0], -1, x.shape[-1])
    if not x.flags.c_contiguous:
        x = x.copy()
    return (x, x_shape)