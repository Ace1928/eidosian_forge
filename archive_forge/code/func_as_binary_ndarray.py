import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def as_binary_ndarray(array, *, variable_name):
    """Return `array` as a numpy.ndarray of dtype bool.

    Raises
    ------
    ValueError:
        An error including the given `variable_name` if `array` can not be
        safely cast to a boolean array.
    """
    array = np.asarray(array)
    if array.dtype != bool:
        if np.any((array != 1) & (array != 0)):
            raise ValueError(f'{variable_name} array is not of dtype boolean or contains values other than 0 and 1 so cannot be safely cast to boolean array.')
    return np.asarray(array, dtype=bool)