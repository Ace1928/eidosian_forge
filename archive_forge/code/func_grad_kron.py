from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def grad_kron(argnum, ans, orig_A, orig_B):
    orig_A_shape = anp.shape(orig_A)
    orig_B_shape = anp.shape(orig_B)

    def vjp(G):
        A, B = (anp.atleast_2d(orig_A), anp.atleast_2d(orig_B))
        shape = list(A.shape + B.shape)
        n = anp.ndim(A)
        shape[n - 1], shape[n] = (shape[n], shape[n - 1])
        reshaped_G = anp.swapaxes(anp.reshape(G, shape), n - 1, n)
        if argnum == 0:
            return match_complex(orig_A, anp.reshape(anp.tensordot(reshaped_G, B, axes=anp.ndim(B)), orig_A_shape))
        else:
            return match_complex(orig_B, anp.reshape(anp.tensordot(A, reshaped_G, axes=anp.ndim(A)), orig_B_shape))
    return vjp