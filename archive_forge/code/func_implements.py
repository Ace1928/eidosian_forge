import numpy as np
import cupy
import cupy.linalg as _cp_linalg
def implements(scipy_func_name):
    """Decorator adds function to the dictionary of implemented functions"""

    def inner(func):
        scipy_func = _scipy_linalg and getattr(_scipy_linalg, scipy_func_name, None)
        if scipy_func:
            _implemented[scipy_func] = func
        else:
            _notfound.append(scipy_func_name)
        return func
    return inner