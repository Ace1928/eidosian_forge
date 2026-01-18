from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def reverse_axis(x, axis):
    x = x.swapaxes(axis, 0)
    x = x[::-1, ...]
    return x.swapaxes(0, axis)