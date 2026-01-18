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
def conditional_dask(func):
    """Conditionally pass dask kwargs to `wrap_xarray_ufunc`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not Dask.dask_flag:
            return func(*args, **kwargs)
        user_kwargs = kwargs.pop('dask_kwargs', None)
        if user_kwargs is None:
            user_kwargs = {}
        default_kwargs = Dask.dask_kwargs
        return func(*args, dask_kwargs={**default_kwargs, **user_kwargs}, **kwargs)
    return wrapper