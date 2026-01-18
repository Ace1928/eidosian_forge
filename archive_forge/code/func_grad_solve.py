from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def grad_solve(argnum, ans, a, b):
    updim = lambda x: x if x.ndim == a.ndim else x[..., None]
    if argnum == 0:
        return lambda g: -_dot(updim(solve(T(a), g)), T(updim(ans)))
    else:
        return lambda g: solve(T(a), g)