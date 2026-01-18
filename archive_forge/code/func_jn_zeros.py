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
def jn_zeros(n, nt):
    """Compute zeros of integer-order Bessel functions Jn.

    Compute `nt` zeros of the Bessel functions :math:`J_n(x)` on the
    interval :math:`(0, \\infty)`. The zeros are returned in ascending
    order. Note that this interval excludes the zero at :math:`x = 0`
    that exists for :math:`n > 0`.

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
    jv: Real-order Bessel functions of the first kind
    jnp_zeros: Zeros of :math:`Jn'`

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    Compute the first four positive roots of :math:`J_3`.

    >>> from scipy.special import jn_zeros
    >>> jn_zeros(3, 4)
    array([ 6.3801619 ,  9.76102313, 13.01520072, 16.22346616])

    Plot :math:`J_3` and its first four positive roots. Note
    that the root located at 0 is not returned by `jn_zeros`.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from scipy.special import jn, jn_zeros
    >>> j3_roots = jn_zeros(3, 4)
    >>> xmax = 18
    >>> xmin = -1
    >>> x = np.linspace(xmin, xmax, 500)
    >>> fig, ax = plt.subplots()
    >>> ax.plot(x, jn(3, x), label=r'$J_3$')
    >>> ax.scatter(j3_roots, np.zeros((4, )), s=30, c='r',
    ...            label=r"$J_3$_Zeros", zorder=5)
    >>> ax.scatter(0, 0, s=30, c='k',
    ...            label=r"Root at 0", zorder=5)
    >>> ax.hlines(0, 0, xmax, color='k')
    >>> ax.set_xlim(xmin, xmax)
    >>> plt.legend()
    >>> plt.show()
    """
    return jnyn_zeros(n, nt)[0]