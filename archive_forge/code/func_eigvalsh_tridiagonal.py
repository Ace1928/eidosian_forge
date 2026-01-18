import warnings
import numpy
from numpy import (array, isfinite, inexact, nonzero, iscomplexobj,
from scipy._lib._util import _asarray_validated
from ._misc import LinAlgError, _datacopied, norm
from .lapack import get_lapack_funcs, _compute_lwork
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def eigvalsh_tridiagonal(d, e, select='a', select_range=None, check_finite=True, tol=0.0, lapack_driver='auto'):
    """
    Solve eigenvalue problem for a real symmetric tridiagonal matrix.

    Find eigenvalues `w` of ``a``::

        a v[:,i] = w[i] v[:,i]
        v.H v    = identity

    For a real symmetric matrix ``a`` with diagonal elements `d` and
    off-diagonal elements `e`.

    Parameters
    ----------
    d : ndarray, shape (ndim,)
        The diagonal elements of the array.
    e : ndarray, shape (ndim-1,)
        The off-diagonal elements of the array.
    select : {'a', 'v', 'i'}, optional
        Which eigenvalues to calculate

        ======  ========================================
        select  calculated
        ======  ========================================
        'a'     All eigenvalues
        'v'     Eigenvalues in the interval (min, max]
        'i'     Eigenvalues with indices min <= i <= max
        ======  ========================================
    select_range : (min, max), optional
        Range of selected eigenvalues
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.
    tol : float
        The absolute tolerance to which each eigenvalue is required
        (only used when ``lapack_driver='stebz'``).
        An eigenvalue (or cluster) is considered to have converged if it
        lies in an interval of this width. If <= 0. (default),
        the value ``eps*|a|`` is used where eps is the machine precision,
        and ``|a|`` is the 1-norm of the matrix ``a``.
    lapack_driver : str
        LAPACK function to use, can be 'auto', 'stemr', 'stebz',  'sterf',
        or 'stev'. When 'auto' (default), it will use 'stemr' if ``select='a'``
        and 'stebz' otherwise. 'sterf' and 'stev' can only be used when
        ``select='a'``.

    Returns
    -------
    w : (M,) ndarray
        The eigenvalues, in ascending order, each repeated according to its
        multiplicity.

    Raises
    ------
    LinAlgError
        If eigenvalue computation does not converge.

    See Also
    --------
    eigh_tridiagonal : eigenvalues and right eiegenvectors for
        symmetric/Hermitian tridiagonal matrices

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import eigvalsh_tridiagonal, eigvalsh
    >>> d = 3*np.ones(4)
    >>> e = -1*np.ones(3)
    >>> w = eigvalsh_tridiagonal(d, e)
    >>> A = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    >>> w2 = eigvalsh(A)  # Verify with other eigenvalue routines
    >>> np.allclose(w - w2, np.zeros(4))
    True
    """
    return eigh_tridiagonal(d, e, eigvals_only=True, select=select, select_range=select_range, check_finite=check_finite, tol=tol, lapack_driver=lapack_driver)