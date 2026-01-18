import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def gaussian_overlap(la, lb, ra, rb, alpha, beta):
    """Compute overlap integral for two primitive Gaussian functions.

    The overlap integral between two Gaussian functions denoted by :math:`a` and :math:`b` can be
    computed as [`Helgaker (1995) p803 <https://www.worldscientific.com/doi/abs/10.1142/9789812832115_0001>`_]:

    .. math::

        S_{ab} = E^{ij} E^{kl} E^{mn} \\left (\\frac{\\pi}{p}  \\right )^{3/2},

    where :math:`E` is a coefficient that can be computed recursively, :math:`i-n` are the angular
    momentum quantum numbers corresponding to different Cartesian components and :math:`p` is
    computed from the exponents of the two Gaussian functions as :math:`p = \\alpha + \\beta`.

    Args:
        la (integer): angular momentum for the first Gaussian function
        lb (integer): angular momentum for the second Gaussian function
        ra (float): position vector of the first Gaussian function
        rb (float): position vector of the second Gaussian function
        alpha (array[float]): exponent of the first Gaussian function
        beta (array[float]): exponent of the second Gaussian function

    Returns:
        array[float]: overlap integral between primitive Gaussian functions

    **Example**

    >>> la, lb = (0, 0, 0), (0, 0, 0)
    >>> ra, rb = np.array([0., 0., 0.]), np.array([0., 0., 0.])
    >>> alpha = np.array([np.pi/2])
    >>> beta = np.array([np.pi/2])
    >>> o = gaussian_overlap(la, lb, ra, rb, alpha, beta)
    >>> o
    array([1.])
    """
    p = alpha + beta
    s = 1.0
    for i in range(3):
        s = s * qml.math.sqrt(np.pi / p) * expansion(la[i], lb[i], ra[i], rb[i], alpha, beta, 0)
    return s