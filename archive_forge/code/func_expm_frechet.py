import numpy as np
import scipy.linalg
def expm_frechet(A, E, method=None, compute_expm=True, check_finite=True):
    """
    Frechet derivative of the matrix exponential of A in the direction E.

    Parameters
    ----------
    A : (N, N) array_like
        Matrix of which to take the matrix exponential.
    E : (N, N) array_like
        Matrix direction in which to take the Frechet derivative.
    method : str, optional
        Choice of algorithm. Should be one of

        - `SPS` (default)
        - `blockEnlarge`

    compute_expm : bool, optional
        Whether to compute also `expm_A` in addition to `expm_frechet_AE`.
        Default is True.
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    expm_A : ndarray
        Matrix exponential of A.
    expm_frechet_AE : ndarray
        Frechet derivative of the matrix exponential of A in the direction E.
    For ``compute_expm = False``, only `expm_frechet_AE` is returned.

    See Also
    --------
    expm : Compute the exponential of a matrix.

    Notes
    -----
    This section describes the available implementations that can be selected
    by the `method` parameter. The default method is *SPS*.

    Method *blockEnlarge* is a naive algorithm.

    Method *SPS* is Scaling-Pade-Squaring [1]_.
    It is a sophisticated implementation which should take
    only about 3/8 as much time as the naive implementation.
    The asymptotics are the same.

    .. versionadded:: 0.13.0

    References
    ----------
    .. [1] Awad H. Al-Mohy and Nicholas J. Higham (2009)
           Computing the Frechet Derivative of the Matrix Exponential,
           with an application to Condition Number Estimation.
           SIAM Journal On Matrix Analysis and Applications.,
           30 (4). pp. 1639-1657. ISSN 1095-7162

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import linalg
    >>> rng = np.random.default_rng()

    >>> A = rng.standard_normal((3, 3))
    >>> E = rng.standard_normal((3, 3))
    >>> expm_A, expm_frechet_AE = linalg.expm_frechet(A, E)
    >>> expm_A.shape, expm_frechet_AE.shape
    ((3, 3), (3, 3))

    Create a 6x6 matrix containing [[A, E], [0, A]]:

    >>> M = np.zeros((6, 6))
    >>> M[:3, :3] = A
    >>> M[:3, 3:] = E
    >>> M[3:, 3:] = A

    >>> expm_M = linalg.expm(M)
    >>> np.allclose(expm_A, expm_M[:3, :3])
    True
    >>> np.allclose(expm_frechet_AE, expm_M[:3, 3:])
    True

    """
    if check_finite:
        A = np.asarray_chkfinite(A)
        E = np.asarray_chkfinite(E)
    else:
        A = np.asarray(A)
        E = np.asarray(E)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError('expected A to be a square matrix')
    if E.ndim != 2 or E.shape[0] != E.shape[1]:
        raise ValueError('expected E to be a square matrix')
    if A.shape != E.shape:
        raise ValueError('expected A and E to be the same shape')
    if method is None:
        method = 'SPS'
    if method == 'SPS':
        expm_A, expm_frechet_AE = expm_frechet_algo_64(A, E)
    elif method == 'blockEnlarge':
        expm_A, expm_frechet_AE = expm_frechet_block_enlarge(A, E)
    else:
        raise ValueError('Unknown implementation %s' % method)
    if compute_expm:
        return (expm_A, expm_frechet_AE)
    else:
        return expm_frechet_AE