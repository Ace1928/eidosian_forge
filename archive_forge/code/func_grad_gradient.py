from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_gradient(ans, x, *vargs, **kwargs):
    axis = kwargs.pop('axis', None)
    if vargs or kwargs:
        raise NotImplementedError('The only optional argument currently supported for np.gradient is axis.')
    if axis is None:
        axis = range(x.ndim)
    elif type(axis) is int:
        axis = [axis]
    else:
        axis = list(axis)
    x_dtype = x.dtype
    x_shape = x.shape
    nd = x.ndim

    def vjp(g):
        if anp.ndim(g) == nd:
            g = g[anp.newaxis]
        out = anp.zeros(x_shape, dtype=x_dtype)
        for i, a in enumerate(axis):
            g_swap = anp.swapaxes(g[i], 0, a)[:, anp.newaxis]
            out_axis = anp.concatenate((-g_swap[0] - 0.5 * g_swap[1], g_swap[0] - 0.5 * g_swap[2], -1.0 * anp.gradient(g_swap, axis=0)[2:-2, 0], 0.5 * g_swap[-3] - g_swap[-1], 0.5 * g_swap[-2] + g_swap[-1]), axis=0)
            out = out + anp.swapaxes(out_axis, 0, a)
        return out
    return vjp