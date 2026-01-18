import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def legroots(c):
    """
    Compute the roots of a Legendre series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * L_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.chebyshev.chebroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such values.
    Roots with multiplicity greater than 1 will also show larger errors as
    the value of the series near such points is relatively insensitive to
    errors in the roots. Isolated roots near the origin can be improved by
    a few iterations of Newton's method.

    The Legendre series basis polynomials aren't powers of ``x`` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import numpy.polynomial.legendre as leg
    >>> leg.legroots((1, 2, 3, 4)) # 4L_3 + 3L_2 + 2L_1 + 1L_0, all real roots
    array([-0.85099543, -0.11407192,  0.51506735]) # may vary

    """
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0] / c[1]])
    m = legcompanion(c)[::-1, ::-1]
    r = la.eigvals(m)
    r.sort()
    return r