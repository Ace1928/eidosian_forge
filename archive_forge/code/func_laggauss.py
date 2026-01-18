import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
def laggauss(deg):
    """
    Gauss-Laguerre quadrature.

    Computes the sample points and weights for Gauss-Laguerre quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[0, \\inf]`
    with the weight function :math:`f(x) = \\exp(-x)`.

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

    The results have only been tested up to degree 100 higher degrees may
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
    m = lagcompanion(c)
    x = la.eigvalsh(m)
    dy = lagval(x, c)
    df = lagval(x, lagder(c))
    x -= dy / df
    fm = lagval(x, c[1:])
    fm /= np.abs(fm).max()
    df /= np.abs(df).max()
    w = 1 / (fm * df)
    w /= w.sum()
    return (x, w)