from __future__ import annotations
from collections import abc
from numbers import Number
import re
from re import Pattern
from typing import TYPE_CHECKING
import numpy as np
from pandas._libs import lib
def is_named_tuple(obj) -> bool:
    """
    Check if the object is a named tuple.

    Parameters
    ----------
    obj : The object to check

    Returns
    -------
    bool
        Whether `obj` is a named tuple.

    Examples
    --------
    >>> from collections import namedtuple
    >>> from pandas.api.types import is_named_tuple
    >>> Point = namedtuple("Point", ["x", "y"])
    >>> p = Point(1, 2)
    >>>
    >>> is_named_tuple(p)
    True
    >>> is_named_tuple((1, 2))
    False
    """
    return isinstance(obj, abc.Sequence) and hasattr(obj, '_fields')