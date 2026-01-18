from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
@unary_to_nary
def make_hvp(fun, x):
    """Builds a function for evaluating the Hessian-vector product at a point,
    which may be useful when evaluating many Hessian-vector products at the same
    point while caching the results of the forward pass."""
    return _make_vjp(grad(fun), x)