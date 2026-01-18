import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def hermite_moment(alpha, beta, t, order, r):
    """Compute the Hermite moment integral recursively.

    The Hermite moment integral in one dimension is defined as

    .. math::

        M_{t}^{e} = \\int_{-\\infty }^{+\\infty} q^e \\Lambda_t dq,

    where :math:`e` is a positive integer, that is represented by the ``order`` argument,
    :math:`q = x, y, z` is the coordinate at which the integral is evaluatedand and
    :math:`\\Lambda_t` is the :math:`t` component of the Hermite Gaussian function. The integral can
    be computed recursively as
    [`Helgaker (1995) p802 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]

    .. math::

        M_{t}^{e+1} = t M_{t-1}^{e} + Q M_{t}^{e} + \\frac{1}{2p} M_{t+1}^{e},

    where :math:`Q` is the distance between the center of the Hermite Gaussian function and the
    origin, at dimension :math:`q = x, y, z` of the Cartesian coordinates system.

    This integral is zero for :math:`t > e` and the base case solution is

    .. math::

        M_t^0 = \\delta _{t0} \\sqrt{\\frac{\\pi}{p}},

    where :math:`p = \\alpha + \\beta` and :math:`\\alpha, \\beta` are the exponents of the Gaussian
    functions that construct the Hermite Gaussian function :math:`\\Lambda`.

    Args:
        alpha (array[float]): exponent of the left Gaussian function
        beta (array[float]): exponent of the right Gaussian function
        t (integer): order of the Hermite Gaussian function
        order (integer): exponent of the position component
        r (array[float]): distance between the center of the Hermite Gaussian function and the origin

    Returns:
        array[float]: the Hermite moment integral

    **Example**

    >>> alpha = np.array([3.42525091])
    >>> beta = np.array([3.42525091])
    >>> t = 0
    >>> order = 1
    >>> r = 1.5
    >>> hermite_moment(alpha, beta, t, order, r)
    array([1.0157925])
    """
    p = alpha + beta
    if t > order or (order == 0 and t != 0):
        return 0.0
    if order == 0 and t == 0:
        return qml.math.sqrt(np.pi / p)
    m = hermite_moment(alpha, beta, t - 1, order - 1, r) * t + hermite_moment(alpha, beta, t, order - 1, r) * r + hermite_moment(alpha, beta, t + 1, order - 1, r) / (2 * p)
    return m