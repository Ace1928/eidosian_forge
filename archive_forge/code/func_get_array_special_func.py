import os
import sys
import functools
import numpy as np
from scipy._lib._array_api import array_namespace, is_cupy, is_torch, is_numpy
from . import _ufuncs
from ._ufuncs import (
def get_array_special_func(f_name, xp, n_array_args):
    if is_numpy(xp):
        f = getattr(_ufuncs, f_name, None)
    elif is_torch(xp):
        f = getattr(xp.special, f_name, None)
    elif is_cupy(xp):
        import cupyx
        f = getattr(cupyx.scipy.special, f_name, None)
    elif xp.__name__ == f'{array_api_compat_prefix}.jax':
        f = getattr(xp.scipy.special, f_name, None)
    else:
        f_scipy = getattr(_ufuncs, f_name, None)

        def f(*args, **kwargs):
            array_args = args[:n_array_args]
            other_args = args[n_array_args:]
            array_args = [np.asarray(arg) for arg in array_args]
            out = f_scipy(*array_args, *other_args, **kwargs)
            return xp.asarray(out)
    return f