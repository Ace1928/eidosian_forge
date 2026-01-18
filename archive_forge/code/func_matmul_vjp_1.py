from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def matmul_vjp_1(ans, A, B):
    A_ndim = anp.ndim(A)
    B_meta = anp.metadata(B)
    return lambda g: matmul_adjoint_1(A, g, A_ndim, B_meta)