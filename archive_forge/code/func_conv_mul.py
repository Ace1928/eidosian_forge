import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def conv_mul(lin_op, rh_val, transpose: bool=False, is_abs: bool=False):
    """Multiply by a convolution operator.

    arameters
    ----------
    lin_op : LinOp
        The root linear operator.
    rh_val : NDArray
        The vector being convolved.
    transpose : bool
        Is the transpose of convolution being applied?
    is_abs : bool
        Is the absolute value of convolution being applied?

    Returns
    -------
    NumPy NDArray
        The convolution.
    """
    constant = mul(lin_op.data, {}, is_abs)
    constant, rh_val = map(intf.from_1D_to_2D, [constant, rh_val])
    if transpose:
        constant = np.flipud(constant)
        return fftconvolve(rh_val, constant, mode='valid')
    elif constant.size >= rh_val.size:
        return fftconvolve(constant, rh_val, mode='full')
    else:
        return fftconvolve(rh_val, constant, mode='full')