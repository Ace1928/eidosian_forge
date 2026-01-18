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
@classmethod
def enable_dask(cls, dask_kwargs=None):
    """To enable Dask.

        Parameters
        ----------
        dask_kwargs : dict
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
        """
    cls.dask_flag = True
    cls.dask_kwargs = dask_kwargs