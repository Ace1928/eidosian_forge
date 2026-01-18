from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def matmul_adjoint_1(A, G, A_ndim, B_meta):
    if anp.ndim(G) == 0:
        return unbroadcast(G * A, B_meta)
    _, B_ndim, _, _ = B_meta
    B_is_vec = B_ndim == 1
    if B_is_vec:
        G = anp.expand_dims(G, anp.ndim(G))
    if A_ndim == 1:
        A = anp.expand_dims(A, 1)
        G = anp.expand_dims(G, anp.ndim(G) - 1)
    else:
        A = anp.swapaxes(A, A_ndim - 2, A_ndim - 1)
    result = anp.matmul(A, G)
    if B_is_vec:
        result = anp.squeeze(result, anp.ndim(G) - 1)
    return unbroadcast(result, B_meta)