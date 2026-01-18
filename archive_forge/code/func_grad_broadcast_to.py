from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_broadcast_to(ans, x, new_shape):
    old_shape = anp.shape(x)
    assert anp.shape(ans) == new_shape
    assert len(old_shape) == len(new_shape), "Can't handle extra leading dims"
    broadcast_axes = tuple(onp.where(onp.logical_and(onp.array(old_shape) == 1, onp.array(new_shape) > 1))[0])
    return lambda g: anp.sum(g, axis=broadcast_axes, keepdims=True)