from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_transpose(ans, x, axes=None):
    if axes is not None:
        axes = anp.argsort(axes)
    return lambda g: anp.transpose(g, axes)