from numpy import asarray_chkfinite, asarray, atleast_2d
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
def cho_factor(a, lower=False, overwrite_a=False, check_finite=True):
    """
    Compute the Cholesky decomposition of a matrix, to use in cho_solve

    Returns a matrix containing the Cholesky decomposition,
    ``A = L L*`` or ``A = U* U`` of a Hermitian positive-definite matrix `a`.
    The return value can be directly used as the first parameter to cho_solve.

    .. warning::
        The returned matrix also contains random data in the entries not
        used by the Cholesky decomposition. If you need to zero these
        entries, use the function `cholesky` instead.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to be decomposed
    lower : bool, optional
        Whether to compute the upper or lower triangular Cholesky factorization
        (Default: upper-triangular)
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance)
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    c : (M, M) ndarray
        Matrix whose upper or lower triangle contains the Cholesky factor
        of `a`. Other parts of the matrix contain random data.
    lower : bool
        Flag indicating whether the factor is in the lower or upper triangle

    Raises
    ------
    LinAlgError
        Raised if decomposition fails.

    See Also
    --------
    cho_solve : Solve a linear set equations using the Cholesky factorization
                of a matrix.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import cho_factor
    >>> A = np.array([[9, 3, 1, 5], [3, 7, 5, 1], [1, 5, 9, 2], [5, 1, 2, 6]])
    >>> c, low = cho_factor(A)
    >>> c
    array([[3.        , 1.        , 0.33333333, 1.66666667],
           [3.        , 2.44948974, 1.90515869, -0.27216553],
           [1.        , 5.        , 2.29330749, 0.8559528 ],
           [5.        , 1.        , 2.        , 1.55418563]])
    >>> np.allclose(np.triu(c).T @ np. triu(c) - A, np.zeros((4, 4)))
    True

    """
    c, lower = _cholesky(a, lower=lower, overwrite_a=overwrite_a, clean=False, check_finite=check_finite)
    return (c, lower)