import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def gaussian_kinetic(la, lb, ra, rb, alpha, beta):
    """Compute the kinetic integral for two primitive Gaussian functions.

    The kinetic integral between two Gaussian functions denoted by :math:`a` and :math:`b` is
    computed as
    [`Helgaker (1995) p805 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        T_{ab} = -\\frac{1}{2} \\left ( D_{ij}^2 D_{kl}^0 D_{mn}^0 + D_{ij}^0 D_{kl}^2 D_{mn}^0 + D_{ij}^0 D_{kl}^0 D_{mn}^2\\right ),

    where :math:`D_{ij}^0 = S_{ij}^0` is an overlap integral and :math:`D_{ij}^2` is computed from
    overlap integrals :math:`S` and the Gaussian exponent :math:`\\beta` as

    .. math::

        D_{ij}^2 = j(j-1)S_{i,j-2}^0 - 2\\beta(2j+1)S_{i,j}^0 + 4\\beta^2 S_{i,j+2}^0.

    Args:
        la (tuple[int]): angular momentum for the first Gaussian function
        lb (tuple[int]): angular momentum for the second Gaussian function
        ra (array[float]): position vector of the first Gaussian function
        rb (array[float]): position vector of the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: kinetic integral between two Gaussian functions

    **Example**

    >>> la, lb = (0, 0, 0), (0, 0, 0)
    >>> ra = np.array([0., 0., 0.])
    >>> rb = rb = np.array([0., 0., 0.])
    >>> alpha = np.array([np.pi/2])
    >>> beta = np.array([np.pi/2])
    >>> t = gaussian_kinetic(la, lb, ra, rb, alpha, beta)
    >>> t
    array([2.35619449])
    """
    p = alpha + beta
    t1 = _diff2(la[0], lb[0], ra[0], rb[0], alpha, beta) * qml.math.sqrt(np.pi / p) * expansion(la[1], lb[1], ra[1], rb[1], alpha, beta, 0) * qml.math.sqrt(np.pi / p) * expansion(la[2], lb[2], ra[2], rb[2], alpha, beta, 0)
    t2 = qml.math.sqrt(np.pi / p) * expansion(la[0], lb[0], ra[0], rb[0], alpha, beta, 0) * _diff2(la[1], lb[1], ra[1], rb[1], alpha, beta) * qml.math.sqrt(np.pi / p) * expansion(la[2], lb[2], ra[2], rb[2], alpha, beta, 0)
    t3 = qml.math.sqrt(np.pi / p) * expansion(la[0], lb[0], ra[0], rb[0], alpha, beta, 0) * qml.math.sqrt(np.pi / p) * expansion(la[1], lb[1], ra[1], rb[1], alpha, beta, 0) * _diff2(la[2], lb[2], ra[2], rb[2], alpha, beta)
    return -0.5 * (t1 + t2 + t3)