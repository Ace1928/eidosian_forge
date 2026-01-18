import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
def r2r_fftpack(forward, x, n=None, axis=-1, norm=None, overwrite_x=False):
    """FFT of a real sequence, returning fftpack half complex format"""
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(None)
    if tmp.dtype.kind == 'c':
        raise TypeError('x must be a real sequence')
    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        raise ValueError(f'invalid number of data points ({tmp.shape[axis]}) specified')
    out = tmp if overwrite_x else None
    return pfft.r2r_fftpack(tmp, (axis,), forward, forward, norm, out, workers)