from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def matmul_vjp_0(ans, A, B):
    A_meta = anp.metadata(A)
    B_ndim = anp.ndim(B)
    return lambda g: matmul_adjoint_0(B, g, A_meta, B_ndim)