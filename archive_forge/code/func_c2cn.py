import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
def c2cn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    """
    Return multidimensional discrete Fourier transform.
    """
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet supported by scipy.fft functions')
    tmp = _asfarray(x)
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    overwrite_x = overwrite_x or _datacopied(tmp, x)
    workers = _workers(workers)
    if len(axes) == 0:
        return x
    tmp, copied = _fix_shape(tmp, shape, axes)
    overwrite_x = overwrite_x or copied
    norm = _normalization(norm, forward)
    out = tmp if overwrite_x and tmp.dtype.kind == 'c' else None
    return pfft.c2c(tmp, axes, forward, norm, out, workers)