import itertools as it
import numpy as np
from scipy.special import factorial2 as fac2
import pennylane as qml
def contracted_norm(l, alpha, a):
    """Compute the normalization constant for a contracted Gaussian function.

    A contracted Gaussian function is defined as

    .. math::

        \\psi = a_1 G_1 + a_2 G_2 + a_3 G_3,

    where :math:`a` denotes the contraction coefficients and :math:`G` is a primitive Gaussian function. The
    normalization constant for this function is computed as

    .. math::

        N(l, \\alpha, a) = [\\frac{\\pi^{3/2}(2l_x-1)!! (2l_y-1)!! (2l_z-1)!!}{2^{l_x + l_y + l_z}}
        \\sum_{i,j} \\frac{a_i a_j}{(\\alpha_i + \\alpha_j)^{{l_x + l_y + l_z+3/2}}}]^{-1/2}

    where :math:`l` and :math:`\\alpha` denote the angular momentum quantum number and the exponent
    of the Gaussian function, respectively.

    Args:
        l (tuple[int]): angular momentum quantum number of the primitive Gaussian functions
        alpha (array[float]): exponents of the primitive Gaussian functions
        a (array[float]): coefficients of the contracted Gaussian functions

    Returns:
        array[float]: normalization coefficient

    **Example**

    >>> l = (0, 0, 0)
    >>> alpha = np.array([3.425250914, 0.6239137298, 0.168855404])
    >>> a = np.array([1.79444183, 0.50032649, 0.18773546])
    >>> n = contracted_norm(l, alpha, a)
    >>> print(n)
    0.39969026908800853
    """
    lx, ly, lz = l
    c = np.pi ** 1.5 / 2 ** sum(l) * _fac2(2 * lx - 1) * _fac2(2 * ly - 1) * _fac2(2 * lz - 1)
    s = (a.reshape(len(a), 1) * a / (alpha.reshape(len(alpha), 1) + alpha) ** (sum(l) + 1.5)).sum()
    n = 1 / qml.math.sqrt(c * s)
    return n