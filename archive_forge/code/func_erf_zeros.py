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
def erf_zeros(nt):
    """Compute the first nt zero in the first quadrant, ordered by absolute value.

    Zeros in the other quadrants can be obtained by using the symmetries
    erf(-z) = erf(z) and erf(conj(z)) = conj(erf(z)).


    Parameters
    ----------
    nt : int
        The number of zeros to compute

    Returns
    -------
    The locations of the zeros of erf : ndarray (complex)
        Complex values at which zeros of erf(z)

    References
    ----------
    .. [1] Zhang, Shanjie and Jin, Jianming. "Computation of Special
           Functions", John Wiley and Sons, 1996.
           https://people.sc.fsu.edu/~jburkardt/f77_src/special_functions/special_functions.html

    Examples
    --------
    >>> from scipy import special
    >>> special.erf_zeros(1)
    array([1.45061616+1.880943j])

    Check that erf is (close to) zero for the value returned by erf_zeros

    >>> special.erf(special.erf_zeros(1))
    array([4.95159469e-14-1.16407394e-16j])

    """
    if floor(nt) != nt or nt <= 0 or (not isscalar(nt)):
        raise ValueError('Argument must be positive scalar integer.')
    return _specfun.cerzo(nt)