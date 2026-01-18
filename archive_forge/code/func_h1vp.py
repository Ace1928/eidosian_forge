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
def h1vp(v, z, n=1):
    """Compute derivatives of Hankel function H1v(z) with respect to `z`.

    Parameters
    ----------
    v : array_like
        Order of Hankel function
    z : array_like
        Argument at which to evaluate the derivative. Can be real or
        complex.
    n : int, default 1
        Order of derivative. For 0 returns the Hankel function `h1v` itself.

    Returns
    -------
    scalar or ndarray
        Values of the derivative of the Hankel function.

    See Also
    --------
    hankel1

    Notes
    -----
    The derivative is computed using the relation DLFM 10.6.7 [2]_.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996, chapter 5.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    .. [2] NIST Digital Library of Mathematical Functions.
           https://dlmf.nist.gov/10.6.E7

    Examples
    --------
    Compute the Hankel function of the first kind of order 0 and
    its first two derivatives at 1.

    >>> from scipy.special import h1vp
    >>> h1vp(0, 1, 0), h1vp(0, 1, 1), h1vp(0, 1, 2)
    ((0.7651976865579664+0.088256964215677j),
     (-0.44005058574493355+0.7812128213002889j),
     (-0.3251471008130329-0.8694697855159659j))

    Compute the first derivative of the Hankel function of the first kind
    for several orders at 1 by providing an array for `v`.

    >>> h1vp([0, 1, 2], 1, 1)
    array([-0.44005059+0.78121282j,  0.3251471 +0.86946979j,
           0.21024362+2.52015239j])

    Compute the first derivative of the Hankel function of the first kind
    of order 0 at several points by providing an array for `z`.

    >>> import numpy as np
    >>> points = np.array([0.5, 1.5, 3.])
    >>> h1vp(0, points, 1)
    array([-0.24226846+1.47147239j, -0.55793651+0.41230863j,
           -0.33905896-0.32467442j])
    """
    n = _nonneg_int_or_fail(n, 'n')
    if n == 0:
        return hankel1(v, z)
    else:
        return _bessel_diff_formula(v, z, n, hankel1, -1)