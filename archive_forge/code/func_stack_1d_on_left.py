from __future__ import (absolute_import, division, print_function)
from functools import reduce
import inspect
import math
import operator
import sys
from pkg_resources import parse_requirements, parse_version
import numpy as np
import pytest
def stack_1d_on_left(x, y):
    """ Stack a 1D array on the left side of a 2D array

    Parameters
    ----------
    x: 1D array
    y: 2D array
        Requirement: ``shape[0] == x.size``
    """
    _x = np.atleast_1d(x)
    _y = np.atleast_1d(y)
    return np.hstack((_x.reshape(_x.size, 1), _y))