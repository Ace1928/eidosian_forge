from warnings import warn
import numpy as np
from numpy import (atleast_2d, arange, zeros_like, imag, diag,
from scipy._lib._util import ComplexWarning
from ._decomp import _asarray_validated
from .lapack import get_lapack_funcs, _compute_lwork
def ldl(A, lower=True, hermitian=True, overwrite_a=False, check_finite=True):
    """ Computes the LDLt or Bunch-Kaufman factorization of a symmetric/
    hermitian matrix.

    This function returns a block diagonal matrix D consisting blocks of size
    at most 2x2 and also a possibly permuted unit lower triangular matrix
    ``L`` such that the factorization ``A = L D L^H`` or ``A = L D L^T``
    holds. If `lower` is False then (again possibly permuted) upper
    triangular matrices are returned as outer factors.

    The permutation array can be used to triangularize the outer factors
    simply by a row shuffle, i.e., ``lu[perm, :]`` is an upper/lower
    triangular matrix. This is also equivalent to multiplication with a
    permutation matrix ``P.dot(lu)``, where ``P`` is a column-permuted
    identity matrix ``I[:, perm]``.

    Depending on the value of the boolean `lower`, only upper or lower
    triangular part of the input array is referenced. Hence, a triangular
    matrix on entry would give the same result as if the full matrix is
    supplied.

    Parameters
    ----------
    A : array_like
        Square input array
    lower : bool, optional
        This switches between the lower and upper triangular outer factors of
        the factorization. Lower triangular (``lower=True``) is the default.
    hermitian : bool, optional
        For complex-valued arrays, this defines whether ``A = A.conj().T`` or
        ``A = A.T`` is assumed. For real-valued arrays, this switch has no
        effect.
    overwrite_a : bool, optional
        Allow overwriting data in `A` (may enhance performance). The default
        is False.
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    lu : ndarray
        The (possibly) permuted upper/lower triangular outer factor of the
        factorization.
    d : ndarray
        The block diagonal multiplier of the factorization.
    perm : ndarray
        The row-permutation index array that brings lu into triangular form.

    Raises
    ------
    ValueError
        If input array is not square.
    ComplexWarning
        If a complex-valued array with nonzero imaginary parts on the
        diagonal is given and hermitian is set to True.

    See Also
    --------
    cholesky, lu

    Notes
    -----
    This function uses ``?SYTRF`` routines for symmetric matrices and
    ``?HETRF`` routines for Hermitian matrices from LAPACK. See [1]_ for
    the algorithm details.

    Depending on the `lower` keyword value, only lower or upper triangular
    part of the input array is referenced. Moreover, this keyword also defines
    the structure of the outer factors of the factorization.

    .. versionadded:: 1.1.0

    References
    ----------
    .. [1] J.R. Bunch, L. Kaufman, Some stable methods for calculating
       inertia and solving symmetric linear systems, Math. Comput. Vol.31,
       1977. :doi:`10.2307/2005787`

    Examples
    --------
    Given an upper triangular array ``a`` that represents the full symmetric
    array with its entries, obtain ``l``, 'd' and the permutation vector `perm`:

    >>> import numpy as np
    >>> from scipy.linalg import ldl
    >>> a = np.array([[2, -1, 3], [0, 2, 0], [0, 0, 1]])
    >>> lu, d, perm = ldl(a, lower=0) # Use the upper part
    >>> lu
    array([[ 0. ,  0. ,  1. ],
           [ 0. ,  1. , -0.5],
           [ 1. ,  1. ,  1.5]])
    >>> d
    array([[-5. ,  0. ,  0. ],
           [ 0. ,  1.5,  0. ],
           [ 0. ,  0. ,  2. ]])
    >>> perm
    array([2, 1, 0])
    >>> lu[perm, :]
    array([[ 1. ,  1. ,  1.5],
           [ 0. ,  1. , -0.5],
           [ 0. ,  0. ,  1. ]])
    >>> lu.dot(d).dot(lu.T)
    array([[ 2., -1.,  3.],
           [-1.,  2.,  0.],
           [ 3.,  0.,  1.]])

    """
    a = atleast_2d(_asarray_validated(A, check_finite=check_finite))
    if a.shape[0] != a.shape[1]:
        raise ValueError('The input array "a" should be square.')
    if a.size == 0:
        return (empty_like(a), empty_like(a), np.array([], dtype=int))
    n = a.shape[0]
    r_or_c = complex if iscomplexobj(a) else float
    if r_or_c is complex and hermitian:
        s, sl = ('hetrf', 'hetrf_lwork')
        if np.any(imag(diag(a))):
            warn('scipy.linalg.ldl():\nThe imaginary parts of the diagonalare ignored. Use "hermitian=False" for factorization ofcomplex symmetric arrays.', ComplexWarning, stacklevel=2)
    else:
        s, sl = ('sytrf', 'sytrf_lwork')
    solver, solver_lwork = get_lapack_funcs((s, sl), (a,))
    lwork = _compute_lwork(solver_lwork, n, lower=lower)
    ldu, piv, info = solver(a, lwork=lwork, lower=lower, overwrite_a=overwrite_a)
    if info < 0:
        raise ValueError(f'{s.upper()} exited with the internal error "illegal value in argument number {-info}". See LAPACK documentation for the error codes.')
    swap_arr, pivot_arr = _ldl_sanitize_ipiv(piv, lower=lower)
    d, lu = _ldl_get_d_and_l(ldu, pivot_arr, lower=lower, hermitian=hermitian)
    lu, perm = _ldl_construct_tri_factor(lu, swap_arr, pivot_arr, lower=lower)
    return (lu, d, perm)