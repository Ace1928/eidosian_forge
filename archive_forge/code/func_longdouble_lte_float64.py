from __future__ import annotations
import warnings
from platform import machine, processor
import numpy as np
from .deprecated import deprecate_with_version
def longdouble_lte_float64():
    """Return True if longdouble appears to have the same precision as float64"""
    return np.longdouble(2 ** 53) == np.longdouble(2 ** 53) + 1