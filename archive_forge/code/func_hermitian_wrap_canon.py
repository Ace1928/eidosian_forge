import numpy as np
from cvxpy.atoms.affine.wraps import skew_symmetric_wrap, symmetric_wrap
from cvxpy.expressions.constants import Constant
def hermitian_wrap_canon(expr, real_args, imag_args, real2imag):
    if imag_args[0] is not None:
        imag_arg = skew_symmetric_wrap(imag_args[0])
    else:
        imag_arg = None
    real_arg = symmetric_wrap(real_args[0])
    return (real_arg, imag_arg)