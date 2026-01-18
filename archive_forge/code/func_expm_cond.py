import numpy as np
import scipy.linalg
def expm_cond(A, check_finite=True):
    """
    Relative condition number of the matrix exponential in the Frobenius norm.

    Parameters
    ----------
    A : 2-D array_like
        Square input matrix with shape (N, N).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    kappa : float
        The relative condition number of the matrix exponential
        in the Frobenius norm

    See Also
    --------
    expm : Compute the exponential of a matrix.
    expm_frechet : Compute the Frechet derivative of the matrix exponential.

    Notes
    -----
    A faster estimate for the condition number in the 1-norm
    has been published but is not yet implemented in SciPy.

    .. versionadded:: 0.14.0

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import expm_cond
    >>> A = np.array([[-0.3, 0.2, 0.6], [0.6, 0.3, -0.1], [-0.7, 1.2, 0.9]])
    >>> k = expm_cond(A)
    >>> k
    1.7787805864469866

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
    else:
        A = np.asarray(A)
    if len(A.shape) != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected a square matrix')
    X = scipy.linalg.expm(A)
    K = expm_frechet_kronform(A, check_finite=False)
    A_norm = scipy.linalg.norm(A, 'fro')
    X_norm = scipy.linalg.norm(X, 'fro')
    K_norm = scipy.linalg.norm(K, 2)
    kappa = K_norm * A_norm / X_norm
    return kappa