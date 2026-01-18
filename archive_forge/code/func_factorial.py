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
def factorial(n, exact=False):
    """
    The factorial of a number or array of numbers.

    The factorial of non-negative integer `n` is the product of all
    positive integers less than or equal to `n`::

        n! = n * (n - 1) * (n - 2) * ... * 1

    Parameters
    ----------
    n : int or array_like of ints
        Input values.  If ``n < 0``, the return value is 0.
    exact : bool, optional
        If True, calculate the answer exactly using long integer arithmetic.
        If False, result is approximated in floating point rapidly using the
        `gamma` function.
        Default is False.

    Returns
    -------
    nf : float or int or ndarray
        Factorial of `n`, as integer or float depending on `exact`.

    Notes
    -----
    For arrays with ``exact=True``, the factorial is computed only once, for
    the largest input, with each other result computed in the process.
    The output dtype is increased to ``int64`` or ``object`` if necessary.

    With ``exact=False`` the factorial is approximated using the gamma
    function:

    .. math:: n! = \\Gamma(n+1)

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import factorial
    >>> arr = np.array([3, 4, 5])
    >>> factorial(arr, exact=False)
    array([   6.,   24.,  120.])
    >>> factorial(arr, exact=True)
    array([  6,  24, 120])
    >>> factorial(5, exact=True)
    120

    """
    if np.ndim(n) == 0 and (not isinstance(n, np.ndarray)):
        if n is None or np.isnan(n):
            return np.nan
        elif not (np.issubdtype(type(n), np.integer) or np.issubdtype(type(n), np.floating)):
            raise ValueError(f'Unsupported datatype for factorial: {type(n)}\nPermitted data types are integers and floating point numbers')
        elif n < 0:
            return 0
        elif exact and np.issubdtype(type(n), np.integer):
            return math.factorial(n)
        return _ufuncs._factorial(n)
    n = asarray(n)
    if n.size == 0:
        return n
    if not (np.issubdtype(n.dtype, np.integer) or np.issubdtype(n.dtype, np.floating)):
        raise ValueError(f'Unsupported datatype for factorial: {n.dtype}\nPermitted data types are integers and floating point numbers')
    if exact and (not np.issubdtype(n.dtype, np.integer)):
        n_flt = n[~np.isnan(n)]
        if np.allclose(n_flt, n_flt.astype(np.int64)):
            warnings.warn('Non-integer arrays (e.g. due to presence of NaNs) together with exact=True are deprecated. Either ensure that the the array has integer dtype or use exact=False.', DeprecationWarning, stacklevel=2)
        else:
            msg = 'factorial with exact=True does not support non-integral arrays'
            raise ValueError(msg)
    if exact:
        return _exact_factorialx_array(n)
    res = _ufuncs._factorial(n)
    if isinstance(n, np.ndarray):
        return np.array(res)
    return res