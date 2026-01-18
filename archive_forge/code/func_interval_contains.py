import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def interval_contains(interval, val):
    """
    Check, inclusively, whether an interval includes a given value.

    Parameters
    ----------
    interval : (float, float)
        The endpoints of the interval.
    val : float
        Value to check is within interval.

    Returns
    -------
    bool
        Whether *val* is within the *interval*.
    """
    a, b = interval
    if a > b:
        a, b = (b, a)
    return a <= val <= b