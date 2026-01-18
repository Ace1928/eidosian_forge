import functools
import importlib
import importlib.resources
import re
import warnings
from functools import lru_cache
import matplotlib.pyplot as plt
import numpy as np
from numpy import newaxis
from .rcparams import rcParams
def conditional_jit(_func=None, **kwargs):
    """Use numba's jit decorator if numba is installed.

    Notes
    -----
        If called without arguments  then return wrapped function.

        @conditional_jit
        def my_func():
            return

        else called with arguments

        @conditional_jit(nopython=True)
        def my_func():
            return

    """
    if _func is None:
        return lambda fn: functools.wraps(fn)(maybe_numba_fn(fn, **kwargs))
    lazy_numba = maybe_numba_fn(_func, **kwargs)
    return functools.wraps(_func)(lazy_numba)