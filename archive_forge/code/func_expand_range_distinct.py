from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def expand_range_distinct(range: TupleFloat2, expand: TupleFloat2 | TupleFloat4=(0, 0, 0, 0), zero_width: float=1) -> TupleFloat2:
    """
    Expand a range with a multiplicative or additive constants

    Similar to :func:`expand_range` but both sides of the range
    expanded using different constants

    Parameters
    ----------
    range : tuple
        Range of data. Size 2
    expand : tuple
        Length 2 or 4. If length is 2, then the same constants
        are used for both sides. If length is 4 then the first
        two are are the Multiplicative (*mul*) and Additive (*add*)
        constants for the lower limit, and the second two are
        the constants for the upper limit.
    zero_width : int | float | timedelta
        Distance to use if range has zero width

    Returns
    -------
    out : tuple
        Expanded range

    Examples
    --------
    >>> expand_range_distinct((3, 8))
    (3, 8)
    >>> expand_range_distinct((0, 10), (0.1, 0))
    (-1.0, 11.0)
    >>> expand_range_distinct((0, 10), (0.1, 0, 0.1, 0))
    (-1.0, 11.0)
    >>> expand_range_distinct((0, 10), (0.1, 0, 0, 0))
    (-1.0, 10)
    >>> expand_range_distinct((0, 10), (0, 2))
    (-2, 12)
    >>> expand_range_distinct((0, 10), (0, 2, 0, 2))
    (-2, 12)
    >>> expand_range_distinct((0, 10), (0, 0, 0, 2))
    (0, 12)
    >>> expand_range_distinct((0, 10), (.1, 2))
    (-3.0, 13.0)
    >>> expand_range_distinct((0, 10), (.1, 2, .1, 2))
    (-3.0, 13.0)
    >>> expand_range_distinct((0, 10), (0, 0, .1, 2))
    (0, 13.0)
    """
    if len(expand) == 2:
        low_mul = high_mul = expand[0]
        low_add = high_add = expand[1]
    else:
        low_mul, low_add, high_mul, high_add = expand
    lower = expand_range(range, low_mul, low_add, zero_width)[0]
    upper = expand_range(range, high_mul, high_add, zero_width)[1]
    return (lower, upper)