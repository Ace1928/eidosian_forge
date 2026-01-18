from __future__ import annotations
import datetime
import sys
import typing
from copy import copy
from typing import overload
import numpy as np
import pandas as pd
from .utils import get_null_value, is_vector
def squish(x: FloatArrayLike, range: TupleFloat2=(0, 1), only_finite: bool=True) -> NDArrayFloat:
    """
    Squish values into range.

    Parameters
    ----------
    x : array_like
        Values that should have out of range values squished.
    range : tuple
        The range onto which to squish the values.
    only_finite: boolean
        When true, only squishes finite values.

    Returns
    -------
    out : array_like
        Values with out of range values squished.

    Examples
    --------
    >>> list(squish([-1.5, 0.2, 0.8, 1.0, 1.2]))
    [0.0, 0.2, 0.8, 1.0, 1.0]

    >>> list(squish([-np.inf, -1.5, 0.2, 0.8, 1.0, np.inf], only_finite=False))
    [0.0, 0.0, 0.2, 0.8, 1.0, 1.0]
    """
    _x = np.asarray(x)
    finite = np.isfinite(_x) if only_finite else True
    _x[np.logical_and(_x < range[0], finite)] = range[0]
    _x[np.logical_and(_x > range[1], finite)] = range[1]
    return _x