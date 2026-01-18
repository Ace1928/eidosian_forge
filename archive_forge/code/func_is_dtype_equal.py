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
def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    if isinstance(target, str):
        if not isinstance(source, str):
            try:
                src = _get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return src == target
            except (TypeError, AttributeError, ImportError):
                return False
    elif isinstance(source, str):
        return is_dtype_equal(target, source)
    try:
        source = _get_dtype(source)
        target = _get_dtype(target)
        return source == target
    except (TypeError, AttributeError, ImportError):
        return False