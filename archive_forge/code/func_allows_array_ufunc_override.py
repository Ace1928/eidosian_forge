from numpy.core.overrides import ARRAY_FUNCTIONS as _array_functions
from numpy import ufunc as _ufunc
import numpy.core.umath as _umath
def allows_array_ufunc_override(func):
    """Determine if a function can be overridden via `__array_ufunc__`

    Parameters
    ----------
    func : callable
        Function that may be overridable via `__array_ufunc__`

    Returns
    -------
    bool
        `True` if `func` is overridable via `__array_ufunc__` and
        `False` otherwise.

    Notes
    -----
    This function is equivalent to ``isinstance(func, np.ufunc)`` and
    will work correctly for ufuncs defined outside of Numpy.

    """
    return isinstance(func, np.ufunc)