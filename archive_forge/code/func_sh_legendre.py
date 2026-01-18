import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def sh_legendre(n, monic=False):
    """Shifted Legendre polynomial.

    Defined as :math:`P^*_n(x) = P_n(2x - 1)` for :math:`P_n` the nth
    Legendre polynomial.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    P : orthopoly1d
        Shifted Legendre polynomial.

    Notes
    -----
    The polynomials :math:`P^*_n` are orthogonal over :math:`[0, 1]`
    with weight function 1.

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        return 0.0 * x + 1.0
    if n == 0:
        return orthopoly1d([], [], 1.0, 1.0, wfunc, (0, 1), monic, lambda x: _ufuncs.eval_sh_legendre(n, x))
    x, w = roots_sh_legendre(n)
    hn = 1.0 / (2 * n + 1.0)
    kn = _gam(2 * n + 1) / _gam(n + 1) ** 2
    p = orthopoly1d(x, w, hn, kn, wfunc, limits=(0, 1), monic=monic, eval_func=lambda x: _ufuncs.eval_sh_legendre(n, x))
    return p