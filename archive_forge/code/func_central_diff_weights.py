from scipy._lib.deprecation import _deprecated
from scipy._lib._finite_differences import _central_diff_weights, _derivative
from numpy import array, frombuffer, load
@_deprecated(msg='scipy.misc.central_diff_weights is deprecated in SciPy v1.10.0; and will be completely removed in SciPy v1.12.0. You may consider using findiff: https://github.com/maroba/findiff or numdifftools: https://github.com/pbrod/numdifftools')
def central_diff_weights(Np, ndiv=1):
    """
    Return weights for an Np-point central derivative.

    Assumes equally-spaced function points.

    If weights are in the vector w, then
    derivative is w[0] * f(x-ho*dx) + ... + w[-1] * f(x+h0*dx)

    .. deprecated:: 1.10.0
        `central_diff_weights` has been deprecated from
        `scipy.misc.central_diff_weights` in SciPy 1.10.0 and
        it will be completely removed in SciPy 1.12.0.
        You may consider using
        findiff: https://github.com/maroba/findiff or
        numdifftools: https://github.com/pbrod/numdifftools

    Parameters
    ----------
    Np : int
        Number of points for the central derivative.
    ndiv : int, optional
        Number of divisions. Default is 1.

    Returns
    -------
    w : ndarray
        Weights for an Np-point central derivative. Its size is `Np`.

    Notes
    -----
    Can be inaccurate for a large number of points.

    Examples
    --------
    We can calculate a derivative value of a function.

    >>> from scipy.misc import central_diff_weights
    >>> def f(x):
    ...     return 2 * x**2 + 3
    >>> x = 3.0 # derivative point
    >>> h = 0.1 # differential step
    >>> Np = 3 # point number for central derivative
    >>> weights = central_diff_weights(Np) # weights for first derivative
    >>> vals = [f(x + (i - Np/2) * h) for i in range(Np)]
    >>> sum(w * v for (w, v) in zip(weights, vals))/h
    11.79999999999998

    This value is close to the analytical solution:
    f'(x) = 4x, so f'(3) = 12

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Finite_difference

    """
    return _central_diff_weights(Np, ndiv)