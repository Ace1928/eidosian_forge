from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
@primitive
def dot_adjoint_1(A, G, A_meta, B_meta):
    _, A_ndim, _, _ = A_meta
    _, B_ndim, B_dtype, _ = B_meta
    needs_transpose = B_ndim > 1 and A_ndim != 0
    swap = (lambda x: onp.swapaxes(x, -1, -2)) if needs_transpose else lambda x: x
    if A_ndim == 0 or A_ndim == 1 or B_ndim == 0:
        contract_num = max(0, A_ndim - (B_ndim != 0))
        out = swap(onp.tensordot(G, A, contract_num))
    else:
        out = swap(onp.tensordot(G, A, [range(-A_ndim - B_ndim + 2, -B_ndim + 1), range(A_ndim - 1)]))
    return onp.asarray(out, dtype=B_dtype)