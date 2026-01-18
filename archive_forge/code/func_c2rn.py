import numpy as np
import functools
from . import pypocketfft as pfft
from .helper import (_asfarray, _init_nd_shape_and_axes, _datacopied,
def c2rn(forward, x, s=None, axes=None, norm=None, overwrite_x=False, workers=None, *, plan=None):
    """Multidimensional inverse discrete fourier transform with real output"""
    if plan is not None:
        raise NotImplementedError('Passing a precomputed plan is not yet supported by scipy.fft functions')
    tmp = _asfarray(x)
    if np.isrealobj(tmp):
        tmp = tmp + 0j
    noshape = s is None
    shape, axes = _init_nd_shape_and_axes(tmp, s, axes)
    if len(axes) == 0:
        raise ValueError('at least 1 axis must be transformed')
    shape = list(shape)
    if noshape:
        shape[-1] = (x.shape[axes[-1]] - 1) * 2
    norm = _normalization(norm, forward)
    workers = _workers(workers)
    lastsize = shape[-1]
    shape[-1] = shape[-1] // 2 + 1
    tmp, _ = tuple(_fix_shape(tmp, shape, axes))
    return pfft.c2r(tmp, axes, lastsize, forward, norm, None, workers)