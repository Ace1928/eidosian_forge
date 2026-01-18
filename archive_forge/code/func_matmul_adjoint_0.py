from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def matmul_adjoint_0(B, G, A_meta, B_ndim):
    if anp.ndim(G) == 0:
        return unbroadcast(G * B, A_meta)
    _, A_ndim, _, _ = A_meta
    if A_ndim == 1:
        G = anp.expand_dims(G, anp.ndim(G) - 1)
    if B_ndim == 1:
        B = anp.expand_dims(B, 0)
        G = anp.expand_dims(G, anp.ndim(G))
    else:
        B = anp.swapaxes(B, B_ndim - 2, B_ndim - 1)
    result = anp.matmul(G, B)
    return unbroadcast(result, A_meta)