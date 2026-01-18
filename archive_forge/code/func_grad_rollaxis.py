from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_rollaxis(ans, a, axis, start=0):
    if axis < 0:
        raise NotImplementedError('Gradient of rollaxis not implemented for axis < 0. Please use moveaxis instead.')
    elif start < 0:
        raise NotImplementedError('Gradient of rollaxis not implemented for start < 0. Please use moveaxis instead.')
    return lambda g: anp.rollaxis(g, start - 1, axis) if start > axis else anp.rollaxis(g, start, axis + 1)