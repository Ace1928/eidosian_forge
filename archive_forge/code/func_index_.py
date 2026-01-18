from __future__ import annotations
import dataclasses
import functools
import hashlib
import os
import subprocess
import sys
from typing import Any, Callable, Final, Iterable, Mapping, TypeVar
from streamlit import env_util
def index_(iterable: Iterable[_Value], x: _Value) -> int:
    """Return zero-based index of the first item whose value is equal to x.
    Raises a ValueError if there is no such item.

    We need a custom implementation instead of the built-in list .index() to
    be compatible with NumPy array and Pandas Series.

    Parameters
    ----------
    iterable : list, tuple, numpy.ndarray, pandas.Series
    x : Any

    Returns
    -------
    int
    """
    for i, value in enumerate(iterable):
        if x == value:
            return i
        elif isinstance(value, float) and isinstance(x, float):
            if abs(x - value) < FLOAT_EQUALITY_EPSILON:
                return i
    raise ValueError(f'{str(x)} is not in iterable')