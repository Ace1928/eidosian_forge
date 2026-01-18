from warnings import warn
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
from scipy.linalg._decomp_qr import qr
from scipy.sparse._sputils import is_pydata_spmatrix
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._interface import IdentityOperator
from scipy.sparse.linalg._onenormest import onenormest
def expm_multiply(A, B, start=None, stop=None, num=None, endpoint=None, traceA=None):
    """
    Compute the action of the matrix exponential of A on B.

    Parameters
    ----------
    A : transposable linear operator
        The operator whose exponential is of interest.
    B : ndarray
        The matrix or vector to be multiplied by the matrix exponential of A.
    start : scalar, optional
        The starting time point of the sequence.
    stop : scalar, optional
        The end time point of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced time points, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of time points to use.
    endpoint : bool, optional
        If True, `stop` is the last time point.  Otherwise, it is not included.
    traceA : scalar, optional
        Trace of `A`. If not given the trace is estimated for linear operators,
        or calculated exactly for sparse matrices. It is used to precondition
        `A`, thus an approximate trace is acceptable.
        For linear operators, `traceA` should be provided to ensure performance
        as the estimation is not guaranteed to be reliable for all cases.

        .. versionadded:: 1.9.0

    Returns
    -------
    expm_A_B : ndarray
         The result of the action :math:`e^{t_k A} B`.

    Warns
    -----
    UserWarning
        If `A` is a linear operator and ``traceA=None`` (default).

    Notes
    -----
    The optional arguments defining the sequence of evenly spaced time points
    are compatible with the arguments of `numpy.linspace`.

    The output ndarray shape is somewhat complicated so I explain it here.
    The ndim of the output could be either 1, 2, or 3.
    It would be 1 if you are computing the expm action on a single vector
    at a single time point.
    It would be 2 if you are computing the expm action on a vector
    at multiple time points, or if you are computing the expm action
    on a matrix at a single time point.
    It would be 3 if you want the action on a matrix with multiple
    columns at multiple time points.
    If multiple time points are requested, expm_A_B[0] will always
    be the action of the expm at the first time point,
    regardless of whether the action is on a vector or a matrix.

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2011)
           "Computing the Action of the Matrix Exponential,
           with an Application to Exponential Integrators."
           SIAM Journal on Scientific Computing,
           33 (2). pp. 488-511. ISSN 1064-8275
           http://eprints.ma.man.ac.uk/1591/

    .. [2] Nicholas J. Higham and Awad H. Al-Mohy (2010)
           "Computing Matrix Functions."
           Acta Numerica,
           19. 159-208. ISSN 0962-4929
           http://eprints.ma.man.ac.uk/1451/

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.sparse import csc_matrix
    >>> from scipy.sparse.linalg import expm, expm_multiply
    >>> A = csc_matrix([[1, 0], [0, 1]])
    >>> A.toarray()
    array([[1, 0],
           [0, 1]], dtype=int64)
    >>> B = np.array([np.exp(-1.), np.exp(-2.)])
    >>> B
    array([ 0.36787944,  0.13533528])
    >>> expm_multiply(A, B, start=1, stop=2, num=3, endpoint=True)
    array([[ 1.        ,  0.36787944],
           [ 1.64872127,  0.60653066],
           [ 2.71828183,  1.        ]])
    >>> expm(A).dot(B)                  # Verify 1st timestep
    array([ 1.        ,  0.36787944])
    >>> expm(1.5*A).dot(B)              # Verify 2nd timestep
    array([ 1.64872127,  0.60653066])
    >>> expm(2*A).dot(B)                # Verify 3rd timestep
    array([ 2.71828183,  1.        ])
    """
    if all((arg is None for arg in (start, stop, num, endpoint))):
        X = _expm_multiply_simple(A, B, traceA=traceA)
    else:
        X, status = _expm_multiply_interval(A, B, start, stop, num, endpoint, traceA=traceA)
    return X