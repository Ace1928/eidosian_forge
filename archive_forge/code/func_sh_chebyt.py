import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def sh_chebyt(n, monic=False):
    """Shifted Chebyshev polynomial of the first kind.

    Defined as :math:`T^*_n(x) = T_n(2x - 1)` for :math:`T_n` the nth
    Chebyshev polynomial of the first kind.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    T : orthopoly1d
        Shifted Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`T^*_n` are orthogonal over :math:`[0, 1]`
    with weight function :math:`(x - x^2)^{-1/2}`.

    """
    base = sh_jacobi(n, 0.0, 0.5, monic=monic)
    if monic:
        return base
    if n > 0:
        factor = 4 ** n / 2.0
    else:
        factor = 1.0
    base._scale(factor)
    return base