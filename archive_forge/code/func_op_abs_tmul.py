import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def op_abs_tmul(lin_op, value):
    """Applies the linear operator |A.T| to the arguments.

    Parameters
    ----------
    lin_op : LinOp
        A linear operator.
    value : NumPy matrix
        A numeric value to apply the operator's transpose to.

    Returns
    -------
    NumPy matrix or SciPy sparse matrix.
        The result of applying the linear operator.
    """
    if lin_op.type is lo.NEG:
        result = value
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {}, True)
        if np.isscalar(coeff):
            result = coeff * value
        else:
            result = coeff.T * value
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {}, True)
        result = value / divisor
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, value, True, True)
    else:
        result = op_tmul(lin_op, value)
    return result