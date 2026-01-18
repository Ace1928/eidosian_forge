import warnings
import numpy as np
from numpy.linalg import inv, LinAlgError, norm, cond, svd
from ._basic import solve, solve_triangular, matrix_balance
from .lapack import get_lapack_funcs
from ._decomp_schur import schur
from ._decomp_lu import lu
from ._decomp_qr import qr
from ._decomp_qz import ordqz
from ._decomp import _asarray_validated
from ._special_matrices import kron, block_diag
def solve_discrete_lyapunov(a, q, method=None):
    """
    Solves the discrete Lyapunov equation :math:`AXA^H - X + Q = 0`.

    Parameters
    ----------
    a, q : (M, M) array_like
        Square matrices corresponding to A and Q in the equation
        above respectively. Must have the same shape.

    method : {'direct', 'bilinear'}, optional
        Type of solver.

        If not given, chosen to be ``direct`` if ``M`` is less than 10 and
        ``bilinear`` otherwise.

    Returns
    -------
    x : ndarray
        Solution to the discrete Lyapunov equation

    See Also
    --------
    solve_continuous_lyapunov : computes the solution to the continuous-time
        Lyapunov equation

    Notes
    -----
    This section describes the available solvers that can be selected by the
    'method' parameter. The default method is *direct* if ``M`` is less than 10
    and ``bilinear`` otherwise.

    Method *direct* uses a direct analytical solution to the discrete Lyapunov
    equation. The algorithm is given in, for example, [1]_. However, it requires
    the linear solution of a system with dimension :math:`M^2` so that
    performance degrades rapidly for even moderately sized matrices.

    Method *bilinear* uses a bilinear transformation to convert the discrete
    Lyapunov equation to a continuous Lyapunov equation :math:`(BX+XB'=-C)`
    where :math:`B=(A-I)(A+I)^{-1}` and
    :math:`C=2(A' + I)^{-1} Q (A + I)^{-1}`. The continuous equation can be
    efficiently solved since it is a special case of a Sylvester equation.
    The transformation algorithm is from Popov (1964) as described in [2]_.

    .. versionadded:: 0.11.0

    References
    ----------
    .. [1] Hamilton, James D. Time Series Analysis, Princeton: Princeton
       University Press, 1994.  265.  Print.
       http://doc1.lbfl.li/aca/FLMF037168.pdf
    .. [2] Gajic, Z., and M.T.J. Qureshi. 2008.
       Lyapunov Matrix Equation in System Stability and Control.
       Dover Books on Engineering Series. Dover Publications.

    Examples
    --------
    Given `a` and `q` solve for `x`:

    >>> import numpy as np
    >>> from scipy import linalg
    >>> a = np.array([[0.2, 0.5],[0.7, -0.9]])
    >>> q = np.eye(2)
    >>> x = linalg.solve_discrete_lyapunov(a, q)
    >>> x
    array([[ 0.70872893,  1.43518822],
           [ 1.43518822, -2.4266315 ]])
    >>> np.allclose(a.dot(x).dot(a.T)-x, -q)
    True

    """
    a = np.asarray(a)
    q = np.asarray(q)
    if method is None:
        if a.shape[0] >= 10:
            method = 'bilinear'
        else:
            method = 'direct'
    meth = method.lower()
    if meth == 'direct':
        x = _solve_discrete_lyapunov_direct(a, q)
    elif meth == 'bilinear':
        x = _solve_discrete_lyapunov_bilinear(a, q)
    else:
        raise ValueError('Unknown solver %s' % method)
    return x