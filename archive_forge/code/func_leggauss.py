import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def leggauss(deg):
    """
    Gauss-Legendre quadrature.

    Computes the sample points and weights for Gauss-Legendre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----

    .. versionadded:: 1.7.0

    The results have only been tested up to degree 100, higher degrees may
    be problematic. The weights are determined by using the fact that

    .. math:: w_k = c / (L'_n(x_k) * L_{n-1}(x_k))

    where :math:`c` is a constant independent of :math:`k` and :math:`x_k`
    is the k'th root of :math:`L_n`, and then scaling the results to get
    the right value when integrating 1.

    """
    ideg = pu._deprecate_as_int(deg, 'deg')
    if ideg <= 0:
        raise ValueError('deg must be a positive integer')
    c = np.array([0] * deg + [1])
    m = legcompanion(c)
    x = la.eigvalsh(m)
    dy = legval(x, c)
    df = legval(x, legder(c))
    x -= dy / df
    fm = legval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)
    w = (w + w[::-1]) / 2
    x = (x - x[::-1]) / 2
    w *= 2.0 / w.sum()
    return (x, w)