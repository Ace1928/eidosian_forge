import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal
@array_function_dispatch(_logn_dispatcher)
def logn(n, x):
    """
    Take log base n of x.

    If `x` contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    n : array_like
       The integer base(s) in which the log is taken.
    x : array_like
       The value(s) whose log base `n` is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The log base `n` of the `x` value(s). If `x` was a scalar, so is
       `out`, otherwise an array is returned.

    Examples
    --------
    >>> np.set_printoptions(precision=4)

    >>> np.emath.logn(2, [4, 8])
    array([2., 3.])
    >>> np.emath.logn(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    x = _fix_real_lt_zero(x)
    n = _fix_real_lt_zero(n)
    return nx.log(x) / nx.log(n)