from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_inner(argnum, ans, A, B):
    A_ndim, B_ndim = (anp.ndim(A), anp.ndim(B))
    if A_ndim == 0 or B_ndim == 0:
        axes = ([], [])
    else:
        axes = ([A_ndim - 1], [B_ndim - 1])
    if argnum == 0:
        return lambda G: tensordot_adjoint_0(B, G, axes, A_ndim, B_ndim)
    elif argnum == 1:
        return lambda G: tensordot_adjoint_1(A, G, axes, A_ndim, B_ndim)