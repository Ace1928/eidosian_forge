import numpy as np
from cvxpy.atoms import bmat
def psd_canon(expr, real_args, imag_args, real2imag):
    """Canonicalize functions that take a Hermitian matrix.
    """
    if imag_args[0] is None:
        matrix = real_args[0]
    else:
        if real_args[0] is None:
            real_args[0] = np.zeros(imag_args[0].shape)
        matrix = bmat([[real_args[0], -imag_args[0]], [imag_args[0], real_args[0]]])
    return ([expr.copy([matrix])], None)