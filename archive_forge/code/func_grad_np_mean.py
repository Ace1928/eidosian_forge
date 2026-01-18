from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_np_mean(ans, x, axis=None, keepdims=False):
    shape, dtype = (anp.shape(x), anp.result_type(x))

    def vjp(g):
        g_repeated, num_reps = repeat_to_match_shape(g, shape, dtype, axis, keepdims)
        return g_repeated / num_reps
    return vjp