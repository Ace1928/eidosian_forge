import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def chebyt(n, monic=False):
    """Chebyshev polynomial of the first kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\\frac{d^2}{dx^2}T_n - x\\frac{d}{dx}T_n + n^2T_n = 0;

    :math:`T_n` is a polynomial of degree :math:`n`.

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
        Chebyshev polynomial of the first kind.

    See Also
    --------
    chebyu : Chebyshev polynomial of the second kind.

    Notes
    -----
    The polynomials :math:`T_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{-1/2}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the first kind of order :math:`n` can
    be obtained as the determinant of specific :math:`n \\times n`
    matrices. As an example we can check how the points obtained from
    the determinant of the following :math:`3 \\times 3` matrix
    lay exactly on :math:`T_3`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    >>> ax.set_title(r'Chebyshev polynomial $T_3$')
    >>> ax.plot(x, chebyt(3)(x), label=rf'$T_3$')
    >>> for p in np.arange(-1.0, 1.0, 0.1):
    ...     ax.plot(p,
    ...             det(np.array([[p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),
    ...             'rx')
    >>> plt.legend(loc='best')
    >>> plt.show()

    They are also related to the Jacobi Polynomials
    :math:`P_n^{(-0.5, -0.5)}` through the relation:

    .. math::
        P_n^{(-0.5, -0.5)}(x) = \\frac{1}{4^n} \\binom{2n}{n} T_n(x)

    Let's verify it for :math:`n = 3`:

    >>> from scipy.special import binom
    >>> from scipy.special import jacobi
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(jacobi(3, -0.5, -0.5)(x),
    ...             1/64 * binom(6, 3) * chebyt(3)(x))
    True

    We can plot the Chebyshev polynomials :math:`T_n` for some values
    of :math:`n`:

    >>> x = np.arange(-1.5, 1.5, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-4.0, 4.0)
    >>> ax.set_title(r'Chebyshev polynomials $T_n$')
    >>> for n in np.arange(2,5):
    ...     ax.plot(x, chebyt(n)(x), label=rf'$T_n={n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    if n < 0:
        raise ValueError('n must be nonnegative.')

    def wfunc(x):
        return 1.0 / sqrt(1 - x * x)
    if n == 0:
        return orthopoly1d([], [], pi, 1.0, wfunc, (-1, 1), monic, lambda x: _ufuncs.eval_chebyt(n, x))
    n1 = n
    x, w, mu = roots_chebyt(n1, mu=True)
    hn = pi / 2
    kn = 2 ** (n - 1)
    p = orthopoly1d(x, w, hn, kn, wfunc, (-1, 1), monic, lambda x: _ufuncs.eval_chebyt(n, x))
    return p