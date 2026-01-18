from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_reshape_list(ans, *arys):
    if len(arys) > 1:
        raise NotImplementedError("Can't handle multiple arguments yet.")
    return lambda g: anp.reshape(g, anp.shape(arys[0]))