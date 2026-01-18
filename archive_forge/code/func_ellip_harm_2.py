import numpy as np
from ._ufuncs import _ellip_harm
from ._ellip_harm_2 import _ellipsoid, _ellipsoid_norm
def ellip_harm_2(h2, k2, n, p, s):
    """
    Ellipsoidal harmonic functions F^p_n(l)

    These are also known as Lame functions of the second kind, and are
    solutions to the Lame equation:

    .. math:: (s^2 - h^2)(s^2 - k^2)F''(s)
              + s(2s^2 - h^2 - k^2)F'(s) + (a - q s^2)F(s) = 0

    where :math:`q = (n+1)n` and :math:`a` is the eigenvalue (not
    returned) corresponding to the solutions.

    Parameters
    ----------
    h2 : float
        ``h**2``
    k2 : float
        ``k**2``; should be larger than ``h**2``
    n : int
        Degree.
    p : int
        Order, can range between [1,2n+1].
    s : float
        Coordinate

    Returns
    -------
    F : float
        The harmonic :math:`F^p_n(s)`

    See Also
    --------
    ellip_harm, ellip_normal

    Notes
    -----
    Lame functions of the second kind are related to the functions of the first kind:

    .. math::

       F^p_n(s)=(2n + 1)E^p_n(s)\\int_{0}^{1/s}
       \\frac{du}{(E^p_n(1/u))^2\\sqrt{(1-u^2k^2)(1-u^2h^2)}}

    .. versionadded:: 0.15.0

    Examples
    --------
    >>> from scipy.special import ellip_harm_2
    >>> w = ellip_harm_2(5,8,2,1,10)
    >>> w
    0.00108056853382

    """
    with np.errstate(all='ignore'):
        return _ellip_harm_2_vec(h2, k2, n, p, s)