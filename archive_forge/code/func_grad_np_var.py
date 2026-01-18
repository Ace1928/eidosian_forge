from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_np_var(ans, x, axis=None, ddof=0, keepdims=False):
    shape, _, dtype, iscomplex = anp.metadata(x)

    def vjp(g):
        if iscomplex:
            g = g + 0j
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        x_minus_mean = anp.conj(x - anp.mean(x, axis=axis, keepdims=True))
        return 2.0 * g_repeated * x_minus_mean / (num_reps - ddof)
    return vjp