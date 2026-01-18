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
def one_de(x):
    """Jitting numpy atleast_1d."""
    if not isinstance(x, np.ndarray):
        return np.atleast_1d(x)
    result = x.reshape(1) if x.ndim == 0 else x
    return result