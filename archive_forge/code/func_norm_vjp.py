from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def norm_vjp(ans, x, ord=None, axis=None):

    def check_implemented():
        matrix_norm = x.ndim == 2 and axis is None or isinstance(axis, tuple)
        if matrix_norm:
            if not (ord is None or ord == 'fro' or ord == 'nuc'):
                raise NotImplementedError('Gradient of matrix norm not implemented for ord={}'.format(ord))
        elif not (ord is None or ord > 1):
            raise NotImplementedError('Gradient of norm not implemented for ord={}'.format(ord))
    if axis is None:
        expand = lambda a: a
    elif isinstance(axis, tuple):
        row_axis, col_axis = axis
        if row_axis > col_axis:
            row_axis = row_axis - 1
        expand = lambda a: anp.expand_dims(anp.expand_dims(a, row_axis), col_axis)
    else:
        expand = lambda a: anp.expand_dims(a, axis=axis)
    if ord == 'nuc':
        if axis is None:
            roll = lambda a: a
            unroll = lambda a: a
        else:
            row_axis, col_axis = axis
            if row_axis > col_axis:
                row_axis = row_axis - 1
            roll = lambda a: anp.rollaxis(anp.rollaxis(a, col_axis, a.ndim), row_axis, a.ndim - 1)
            unroll = lambda a: anp.rollaxis(anp.rollaxis(a, a.ndim - 2, row_axis), a.ndim - 1, col_axis)
    check_implemented()

    def vjp(g):
        if ord in (None, 2, 'fro'):
            return expand(g / ans) * x
        elif ord == 'nuc':
            x_rolled = roll(x)
            u, s, vt = svd(x_rolled, full_matrices=False)
            uvt_rolled = _dot(u, vt)
            uvt = unroll(uvt_rolled)
            g = expand(g)
            return g * uvt
        else:
            return expand(g / ans ** (ord - 1)) * x * anp.abs(x) ** (ord - 2)
    return vjp