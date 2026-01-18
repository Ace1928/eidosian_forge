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
def euler(n):
    """Euler numbers E(0), E(1), ..., E(n).

    The Euler numbers [1]_ are also known as the secant numbers.

    Because ``euler(n)`` returns floating point values, it does not give
    exact values for large `n`.  The first inexact value is E(22).

    Parameters
    ----------
    n : int
        The highest index of the Euler number to be returned.

    Returns
    -------
    ndarray
        The Euler numbers [E(0), E(1), ..., E(n)].
        The odd Euler numbers, which are all zero, are included.

    References
    ----------
    .. [1] Sequence A122045, The On-Line Encyclopedia of Integer Sequences,
           https://oeis.org/A122045
    .. [2] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import euler
    >>> euler(6)
    array([  1.,   0.,  -1.,   0.,   5.,   0., -61.])

    >>> euler(13).astype(np.int64)
    array([      1,       0,      -1,       0,       5,       0,     -61,
                 0,    1385,       0,  -50521,       0, 2702765,       0])

    >>> euler(22)[-1]  # Exact value of E(22) is -69348874393137901.
    -69348874393137976.0

    """
    if not isscalar(n) or n < 0:
        raise ValueError('n must be a non-negative integer.')
    n = int(n)
    if n < 2:
        n1 = 2
    else:
        n1 = n
    return _specfun.eulerb(n1)[:n + 1]