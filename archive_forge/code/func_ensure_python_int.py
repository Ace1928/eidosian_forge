from __future__ import annotations
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    if not (is_integer(value) or is_float(value)):
        if not is_scalar(value):
            raise TypeError(f'Value needs to be a scalar value, was type {type(value).__name__}')
        raise TypeError(f'Wrong type {type(value)} for value {value}')
    try:
        new_value = int(value)
        assert new_value == value
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f'Wrong type {type(value)} for value {value}') from err
    return new_value