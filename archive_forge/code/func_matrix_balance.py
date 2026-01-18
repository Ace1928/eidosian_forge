from warnings import warn
from itertools import product
import numpy as np
from numpy import atleast_1d, atleast_2d
from .lapack import get_lapack_funcs, _compute_lwork
from ._misc import LinAlgError, _datacopied, LinAlgWarning
from ._decomp import _asarray_validated
from . import _decomp, _decomp_svd
from ._solve_toeplitz import levinson
from ._cythonized_array_utils import find_det_from_lu
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
from scipy.linalg._flinalg_py import get_flinalg_funcs  # noqa: F401
def matrix_balance(A, permute=True, scale=True, separate=False, overwrite_a=False):
    """
    Compute a diagonal similarity transformation for row/column balancing.

    The balancing tries to equalize the row and column 1-norms by applying
    a similarity transformation such that the magnitude variation of the
    matrix entries is reflected to the scaling matrices.

    Moreover, if enabled, the matrix is first permuted to isolate the upper
    triangular parts of the matrix and, again if scaling is also enabled,
    only the remaining subblocks are subjected to scaling.

    The balanced matrix satisfies the following equality

    .. math::

                        B = T^{-1} A T

    The scaling coefficients are approximated to the nearest power of 2
    to avoid round-off errors.

    Parameters
    ----------
    A : (n, n) array_like
        Square data matrix for the balancing.
    permute : bool, optional
        The selector to define whether permutation of A is also performed
        prior to scaling.
    scale : bool, optional
        The selector to turn on and off the scaling. If False, the matrix
        will not be scaled.
    separate : bool, optional
        This switches from returning a full matrix of the transformation
        to a tuple of two separate 1-D permutation and scaling arrays.
    overwrite_a : bool, optional
        This is passed to xGEBAL directly. Essentially, overwrites the result
        to the data. It might increase the space efficiency. See LAPACK manual
        for details. This is False by default.

    Returns
    -------
    B : (n, n) ndarray
        Balanced matrix
    T : (n, n) ndarray
        A possibly permuted diagonal matrix whose nonzero entries are
        integer powers of 2 to avoid numerical truncation errors.
    scale, perm : (n,) ndarray
        If ``separate`` keyword is set to True then instead of the array
        ``T`` above, the scaling and the permutation vectors are given
        separately as a tuple without allocating the full array ``T``.

    Notes
    -----
    This algorithm is particularly useful for eigenvalue and matrix
    decompositions and in many cases it is already called by various
    LAPACK routines.

    The algorithm is based on the well-known technique of [1]_ and has
    been modified to account for special cases. See [2]_ for details
    which have been implemented since LAPACK v3.5.0. Before this version
    there are corner cases where balancing can actually worsen the
    conditioning. See [3]_ for such examples.

    The code is a wrapper around LAPACK's xGEBAL routine family for matrix
    balancing.

    .. versionadded:: 0.19.0

    References
    ----------
    .. [1] B.N. Parlett and C. Reinsch, "Balancing a Matrix for
       Calculation of Eigenvalues and Eigenvectors", Numerische Mathematik,
       Vol.13(4), 1969, :doi:`10.1007/BF02165404`
    .. [2] R. James, J. Langou, B.R. Lowery, "On matrix balancing and
       eigenvector computation", 2014, :arxiv:`1401.5766`
    .. [3] D.S. Watkins. A case where balancing is harmful.
       Electron. Trans. Numer. Anal, Vol.23, 2006.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> x = np.array([[1,2,0], [9,1,0.01], [1,2,10*np.pi]])

    >>> y, permscale = linalg.matrix_balance(x)
    >>> np.abs(x).sum(axis=0) / np.abs(x).sum(axis=1)
    array([ 3.66666667,  0.4995005 ,  0.91312162])

    >>> np.abs(y).sum(axis=0) / np.abs(y).sum(axis=1)
    array([ 1.2       ,  1.27041742,  0.92658316])  # may vary

    >>> permscale  # only powers of 2 (0.5 == 2^(-1))
    array([[  0.5,   0. ,  0. ],  # may vary
           [  0. ,   1. ,  0. ],
           [  0. ,   0. ,  1. ]])

    """
    A = np.atleast_2d(_asarray_validated(A, check_finite=True))
    if not np.equal(*A.shape):
        raise ValueError('The data matrix for balancing should be square.')
    gebal = get_lapack_funcs('gebal', (A,))
    B, lo, hi, ps, info = gebal(A, scale=scale, permute=permute, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError(f'xGEBAL exited with the internal error "illegal value in argument number {-info}.". See LAPACK documentation for the xGEBAL error codes.')
    scaling = np.ones_like(ps, dtype=float)
    scaling[lo:hi + 1] = ps[lo:hi + 1]
    ps = ps.astype(int, copy=False) - 1
    n = A.shape[0]
    perm = np.arange(n)
    if hi < n:
        for ind, x in enumerate(ps[hi + 1:][::-1], 1):
            if n - ind == x:
                continue
            perm[[x, n - ind]] = perm[[n - ind, x]]
    if lo > 0:
        for ind, x in enumerate(ps[:lo]):
            if ind == x:
                continue
            perm[[x, ind]] = perm[[ind, x]]
    if separate:
        return (B, (scaling, perm))
    iperm = np.empty_like(perm)
    iperm[perm] = np.arange(n)
    return (B, np.diag(scaling)[iperm, :])