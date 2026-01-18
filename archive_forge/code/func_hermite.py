import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def hermite(n, monic=False):
    """Physicist's Hermite polynomial.

    Defined by

    .. math::

        H_n(x) = (-1)^ne^{x^2}\\frac{d^n}{dx^n}e^{-x^2};

    :math:`H_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    H : orthopoly1d
        Hermite polynomial.

    Notes
    -----
    The polynomials :math:`H_n` are orthogonal over :math:`(-\\infty,
    \\infty)` with weight function :math:`e^{-x^2}`.

    Examples
    --------
    >>> from scipy import special
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> p_monic = special.hermite(3, monic=True)
    >>> p_monic
    poly1d([ 1. ,  0. , -1.5,  0. ])
    >>> p_monic(1)
    -0.49999999999999983
    >>> x = np.linspace(-3, 3, 400)
    >>> y = p_monic(x)
    >>> plt.plot(x, y)
    >>> plt.title("Monic Hermite polynomial of degree 3")
    >>> plt.xlabel("x")
    >>> plt.ylabel("H_3(x)")
    >>> plt.show()

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_hermite(n1)

    def wfunc(x):
        return exp(-x * x)
    if n == 0:
        x, w = ([], [])
    hn = 2 ** n * _gam(n + 1) * sqrt(pi)
    kn = 2 ** n
    p = orthopoly1d(x, w, hn, kn, wfunc, (-inf, inf), monic, lambda x: _ufuncs.eval_hermite(n, x))
    return p