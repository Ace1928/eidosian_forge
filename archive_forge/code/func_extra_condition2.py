from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def extra_condition2(alpha, phi):
    if gval_alpha[0] != alpha:
        derphi(alpha)
    x = xk + alpha * pk
    return extra_condition(alpha, x, phi, gval[0])