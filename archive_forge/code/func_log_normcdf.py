import numpy as np
import scipy.sparse
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.sum import sum as sum_
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.expressions.expression import Expression
def log_normcdf(x):
    """Elementwise log of the cumulative distribution function of a standard normal random variable.

    The implementation is a quadratic approximation with modest accuracy over [-4, 4].
    For details on the nature of the approximation, refer to
    `CVXPY GitHub PR #1224 <https://github.com/cvxpy/cvxpy/pull/1224#issue-793221374>`_.

    .. note::

        SciPy's analog of ``log_normcdf`` is called `log_ndtr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.log_ndtr.html>`_.
        We opted not to use that name because its meaning would not be obvious to the casual user.
    """
    A = scipy.sparse.diags(np.sqrt([0.02301291, 0.08070214, 0.16411522, 0.09003495, 0.08200854, 0.01371543, 0.04641081]))
    b = np.array([[3.0, 2.0, 1.0, 0.0, -1.0, -2.5, -3.5]]).reshape(-1, 1)
    x = Expression.cast_to_const(x)
    flat_x = reshape(x, (1, x.size))
    y = A @ (b @ np.ones(flat_x.shape) - np.ones(b.shape) @ flat_x)
    out = -sum_(maximum(y, 0) ** 2, axis=0)
    return reshape(out, x.shape)