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
def bernoulli(n):
    """Bernoulli numbers B0..Bn (inclusive).

    Parameters
    ----------
    n : int
        Indicated the number of terms in the Bernoulli series to generate.

    Returns
    -------
    ndarray
        The Bernoulli numbers ``[B(0), B(1), ..., B(n)]``.

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html
    .. [2] "Bernoulli number", Wikipedia, https://en.wikipedia.org/wiki/Bernoulli_number

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import bernoulli, zeta
    >>> bernoulli(4)
    array([ 1.        , -0.5       ,  0.16666667,  0.        , -0.03333333])

    The Wikipedia article ([2]_) points out the relationship between the
    Bernoulli numbers and the zeta function, ``B_n^+ = -n * zeta(1 - n)``
    for ``n > 0``:

    >>> n = np.arange(1, 5)
    >>> -n * zeta(1 - n)
    array([ 0.5       ,  0.16666667, -0.        , -0.03333333])

    Note that, in the notation used in the wikipedia article,
    `bernoulli` computes ``B_n^-`` (i.e. it used the convention that
    ``B_1`` is -1/2).  The relation given above is for ``B_n^+``, so the
    sign of 0.5 does not match the output of ``bernoulli(4)``.

    """
    if not isscalar(n) or n < 0:
        raise ValueError('n must be a non-negative integer.')
    n = int(n)
    if n < 2:
        n1 = 2
    else:
        n1 = n
    return _specfun.bernob(int(n1))[:n + 1]