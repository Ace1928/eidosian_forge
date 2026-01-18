import copy
import numpy as np
from scipy.signal import fftconvolve
import cvxpy.interface as intf
import cvxpy.lin_ops.lin_op as lo
def op_tmul(lin_op, value):
    """Applies the transpose of the linear operator to the arguments.

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
    if lin_op.type is lo.SUM:
        result = value
    elif lin_op.type is lo.NEG:
        result = -value
    elif lin_op.type is lo.MUL:
        coeff = mul(lin_op.data, {})
        if np.isscalar(coeff):
            result = coeff * value
        else:
            result = coeff.T * value
    elif lin_op.type is lo.DIV:
        divisor = mul(lin_op.data, {})
        result = value / divisor
    elif lin_op.type is lo.SUM_ENTRIES:
        result = np.asmatrix(np.ones(lin_op.args[0].shape)) * value
    elif lin_op.type is lo.INDEX:
        row_slc, col_slc = lin_op.data
        result = np.asmatrix(np.zeros(lin_op.args[0].shape))
        result[row_slc, col_slc] = value
    elif lin_op.type is lo.TRANSPOSE:
        result = value.T
    elif lin_op.type is lo.PROMOTE:
        result = np.ones(lin_op.shape[0]).dot(value)
    elif lin_op.type is lo.DIAG_VEC:
        result = np.diag(value)
        if isinstance(result, np.matrix):
            result = result.A[0]
    elif lin_op.type is lo.CONV:
        result = conv_mul(lin_op, value, transpose=True)
    else:
        raise Exception('Unknown linear operator.')
    return result