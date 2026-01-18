import sys
import types
import inspect
from functools import wraps, update_wrapper
from sympy.utilities.exceptions import sympy_deprecation_warning
def conserve_mpmath_dps(func):
    """After the function finishes, resets the value of mpmath.mp.dps to
    the value it had before the function was run."""
    import mpmath

    def func_wrapper(*args, **kwargs):
        dps = mpmath.mp.dps
        try:
            return func(*args, **kwargs)
        finally:
            mpmath.mp.dps = dps
    func_wrapper = update_wrapper(func_wrapper, func)
    return func_wrapper