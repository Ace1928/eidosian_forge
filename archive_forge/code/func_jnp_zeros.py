import operator
import numpy as np
import math
import warnings
from collections import defaultdict
from heapq import heapify, heappop
from numpy import (pi, asarray, floor, isscalar, iscomplex, real,
from . import _ufuncs
from ._ufuncs import (mathieu_a, mathieu_b, iv, jv, gamma,
from . import _specfun
from ._comb import _comb_int
from scipy._lib.deprecation import _NoValue, _deprecate_positional_args
def jnp_zeros(n, nt):
    """Compute zeros of integer-order Bessel function derivatives Jn'.

    Compute `nt` zeros of the functions :math:`J_n'(x)` on the
    interval :math:`(0, \\infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 1`.

    Parameters
    ----------
    n : int
        Order of Bessel function
    nt : int
        Number of zeros to return

    Returns
    -------
    ndarray
        First `nt` zeros of the Bessel function.

    See Also
    --------
    jvp: Derivatives of integer-order Bessel functions of the first kind
    jv: Float-order Bessel functions of the first kind

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four roots of :math:`J_2'`.

    >>> from scipy.special import jnp_zeros
    >>> jnp_zeros(2, 4)
    array([ 3.05423693,  6.70613319,  9.96946782, 13.17037086])

    As `jnp_zeros` yields the roots of :math:`J_n'`, it can be used to
    compute the locations of the peaks of :math:`J_n`. Plot
    :math:`J_2`, :math:`J_2'` and the locations of the roots of :math:`J_2'`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jn, jnp_zeros, jvp
    >>> j2_roots = jnp_zeros(2, 4)
    >>> xmax = 15
    >>> x = np.linspace(0, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jn(2, x), label=r'$J_2$')
    >>> ax.plot(x, jvp(2, x, 1), label=r"$J_2'$")
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.scatter(j2_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"Roots of $J_2'$", zorder=5)
    >>> ax.set_ylim(-0.4, 0.8)
    >>> ax.set_xlim(0, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[1]