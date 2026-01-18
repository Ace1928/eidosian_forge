import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
def c2c(forward, x, n=None, axis=-1, norm=None, overwrite_x=False, workers=None, *, plan=None):
    """ Return discrete Fourier transform of real or complex sequence. """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet supported by scipy.fft functions')
    tmp = _asfarray(x)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    norm = _normalization(norm, forward)
    workers = _workers(workers)
    if n is not None:
        tmp, copied = _fix_shape_1d(tmp, n, axis)
        overwrite_x = overwrite_x or copied
    elif tmp.shape[axis] < 1:
        message = f'invalid number of data points ({tmp.shape[axis]}) specified'
        raise ValueError(message)
    out = tmp if overwrite_x and tmp.dtype.kind == 'c' else None
    return pfft.c2c(tmp, (axis,), forward, norm, out, workers)