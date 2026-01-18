from __future__ import absolute_import
from functools import partial
import numpy.linalg as npla
from .numpy_wrapper import wrap_namespace
from . import numpy_wrapper as anp
from autograd.extend import defvjp, defjvp
def conjugate_solve(L, X):
    return solve_trans(L, T(solve_trans(L, T(X))))