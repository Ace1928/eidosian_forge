import numpy as np
from numpy import roll, newaxis
def frt2(a):
    """Compute the 2-dimensional finite Radon transform (FRT) for the input array.

    Parameters
    ----------
    a : ndarray of int, shape (M, M)
        Input array.

    Returns
    -------
    FRT : ndarray of int, shape (M+1, M)
        Finite Radon Transform array of coefficients.

    See Also
    --------
    ifrt2 : The two-dimensional inverse FRT.

    Notes
    -----
    The FRT has a unique inverse if and only if M is prime. [FRT]
    The idea for this algorithm is due to Vlad Negnevitski.

    Examples
    --------

    Generate a test image:
    Use a prime number for the array dimensions

    >>> SIZE = 59
    >>> img = np.tri(SIZE, dtype=np.int32)

    Apply the Finite Radon Transform:

    >>> f = frt2(img)

    References
    ----------
    .. [FRT] A. Kingston and I. Svalbe, "Projective transforms on periodic
             discrete image arrays," in P. Hawkes (Ed), Advances in Imaging
             and Electron Physics, 139 (2006)

    """
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Input must be a square, 2-D array')
    ai = a.copy()
    n = ai.shape[0]
    f = np.empty((n + 1, n), np.uint32)
    f[0] = ai.sum(axis=0)
    for m in range(1, n):
        for row in range(1, n):
            ai[row] = roll(ai[row], -row)
        f[m] = ai.sum(axis=0)
    f[n] = ai.sum(axis=1)
    return f