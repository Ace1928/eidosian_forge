from __future__ import absolute_import
from future.utils import string_types
from functools import partial
import numpy as onp
from ..util import func
from . import numpy_wrapper as anp
from .numpy_boxes import ArrayBox
from autograd.extend import (primitive, vspace, defvjp, defvjp_argnum,
def dot_vjp_1(ans, A, B):
    A_meta, B_meta = (anp.metadata(A), anp.metadata(B))
    return lambda g: match_complex(B, dot_adjoint_1(A, g, A_meta, B_meta))