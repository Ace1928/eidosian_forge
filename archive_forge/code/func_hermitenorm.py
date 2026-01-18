import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def hermitenorm(n, monic=False):
    """Normalized (probabilist's) Hermite polynomial.

    Defined by

    .. math::

        He_n(x) = (-1)^ne^{x^2/2}\\frac{d^n}{dx^n}e^{-x^2/2};

    :math:`He_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    He : orthopoly1d
        Hermite polynomial.

    Notes
    -----

    The polynomials :math:`He_n` are orthogonal over :math:`(-\\infty,
    \\infty)` with weight function :math:`e^{-x^2/2}`.

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermitenorm(n1)

    def wfunc(x):
        return exp(-x * x / 2.0)
    if n == 0:
        x, w = ([], [])
    hn = sqrt(2 * pi) * _gam(n + 1)
    kn = 1.0
    p = orthopoly1d(x, w, hn, kn, wfunc=wfunc, limits=(-inf, inf), monic=monic, eval_func=lambda x: _ufuncs.eval_hermitenorm(n, x))
    return p