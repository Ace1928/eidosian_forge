from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_np_cumsum(ans, x, axis=None):

    def vjp(g):
        if axis:
            return reverse_axis(anp.cumsum(reverse_axis(g, axis), axis), axis)
        else:
            return anp.reshape(anp.cumsum(g[::-1], axis)[::-1], x.shape)
    return vjp