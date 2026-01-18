import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def genlaguerre(n, alpha, monic=False):
    """Generalized (associated) Laguerre polynomial.

    Defined to be the solution of

    .. math::
        x\\frac{d^2}{dx^2}L_n^{(\\alpha)}
          + (\\alpha + 1 - x)\\frac{d}{dx}L_n^{(\\alpha)}
          + nL_n^{(\\alpha)} = 0,

    where :math:`\\alpha > -1`; :math:`L_n^{(\\alpha)}` is a polynomial
    of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    alpha : float
        Parameter, must be greater than -1.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    L : orthopoly1d
        Generalized Laguerre polynomial.

    See Also
    --------
    laguerre : Laguerre polynomial.
    hyp1f1 : confluent hypergeometric function

    Notes
    -----
    For fixed :math:`\\alpha`, the polynomials :math:`L_n^{(\\alpha)}`
    are orthogonal over :math:`[0, \\infty)` with weight function
    :math:`e^{-x}x^\\alpha`.

    The Laguerre polynomials are the special case where :math:`\\alpha
    = 0`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    The generalized Laguerre polynomials are closely related to the confluent
    hypergeometric function :math:`{}_1F_1`:

        .. math::
            L_n^{(\\alpha)} = \\binom{n + \\alpha}{n} {}_1F_1(-n, \\alpha +1, x)

    This can be verified, for example,  for :math:`n = \\alpha = 3` over the
    interval :math:`[-1, 1]`:

    >>> import numpy as np
    >>> from scipy.special import binom
    >>> from scipy.special import genlaguerre
    >>> from scipy.special import hyp1f1
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(genlaguerre(3, 3)(x), binom(6, 3) * hyp1f1(-3, 4, x))
    True

    This is the plot of the generalized Laguerre polynomials
    :math:`L_3^{(\\alpha)}` for some values of :math:`\\alpha`:

    >>> import matplotlib.pyplot as plt
    >>> x = np.arange(-4.0, 12.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-5.0, 10.0)
    >>> ax.set_title(r'Generalized Laguerre polynomials $L_3^{\\alpha}$')
    >>> for alpha in np.arange(0, 5):
    ...     ax.plot(x, genlaguerre(3, alpha)(x), label=rf'$L_3^{(alpha)}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if alpha <= -1:
        raise ValueError('alpha must be > -1')
    if n < 0:
        raise ValueError('n must be nonnegative.')
    if n == 0:
        n1 = n + 1
    else:
        n1 = n
    x, w = roots_genlaguerre(n1, alpha)

    def wfunc(x):
        return exp(-x) * x ** alpha
    if n == 0:
        x, w = ([], [])
    hn = _gam(n + alpha + 1) / _gam(n + 1)
    kn = (-1) ** n / _gam(n + 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (0, inf), monic, lambda x: _ufuncs.eval_genlaguerre(n, alpha, x))
    return p