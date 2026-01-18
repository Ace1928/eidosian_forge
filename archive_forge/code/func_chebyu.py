import numpy as np
from numpy import (exp, inf, pi, sqrt, floor, sin, cos, around,
from scipy import linalg
from scipy.special import airy
from . import _specfun  # type: ignore
from . import _ufuncs
def chebyu(n, monic=False):
    """Chebyshev polynomial of the second kind.

    Defined to be the solution of

    .. math::
        (1 - x^2)\\frac{d^2}{dx^2}U_n - 3x\\frac{d}{dx}U_n
          + n(n + 2)U_n = 0;

    :math:`U_n` is a polynomial of degree :math:`n`.

    Parameters
    ----------
    n : int
        Degree of the polynomial.
    monic : bool, optional
        If `True`, scale the leading coefficient to be 1. Default is
        `False`.

    Returns
    -------
    U : orthopoly1d
        Chebyshev polynomial of the second kind.

    See Also
    --------
    chebyt : Chebyshev polynomial of the first kind.

    Notes
    -----
    The polynomials :math:`U_n` are orthogonal over :math:`[-1, 1]`
    with weight function :math:`(1 - x^2)^{1/2}`.

    References
    ----------
    .. [AS] Milton Abramowitz and Irene A. Stegun, eds.
        Handbook of Mathematical Functions with Formulas,
        Graphs, and Mathematical Tables. New York: Dover, 1972.

    Examples
    --------
    Chebyshev polynomials of the second kind of order :math:`n` can
    be obtained as the determinant of specific :math:`n \\times n`
    matrices. As an example we can check how the points obtained from
    the determinant of the following :math:`3 \\times 3` matrix
    lay exactly on :math:`U_3`:

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.linalg import det
    >>> from scipy.special import chebyu
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-2.0, 2.0)
    >>> ax.set_title(r'Chebyshev polynomial $U_3$')
    >>> ax.plot(x, chebyu(3)(x), label=rf'$U_3$')
    >>> for p in np.arange(-1.0, 1.0, 0.1):
    ...     ax.plot(p,
    ...             det(np.array([[2*p, 1, 0], [1, 2*p, 1], [0, 1, 2*p]])),
    ...             'rx')
    >>> plt.legend(loc='best')
    >>> plt.show()

    They satisfy the recurrence relation:

    .. math::
        U_{2n-1}(x) = 2 T_n(x)U_{n-1}(x)

    where the :math:`T_n` are the Chebyshev polynomial of the first kind.
    Let's verify it for :math:`n = 2`:

    >>> from scipy.special import chebyt
    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> np.allclose(chebyu(3)(x), 2 * chebyt(2)(x) * chebyu(1)(x))
    True

    We can plot the Chebyshev polynomials :math:`U_n` for some values
    of :math:`n`:

    >>> x = np.arange(-1.0, 1.0, 0.01)
    >>> fig, ax = plt.subplots()
    >>> ax.set_ylim(-1.5, 1.5)
    >>> ax.set_title(r'Chebyshev polynomials $U_n$')
    >>> for n in np.arange(1,5):
    ...     ax.plot(x, chebyu(n)(x), label=rf'$U_n={n}$')
    >>> plt.legend(loc='best')
    >>> plt.show()

    """
    base = jacobi(n, 0.5, 0.5, monic=monic)
    if monic:
        return base
    factor = sqrt(pi) / 2.0 * _gam(n + 2) / _gam(n + 1.5)
    base._scale(factor)
    return base