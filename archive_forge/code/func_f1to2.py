import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def f1to2(x):
    """produces an asymmetric non-square matrix from x"""
    assert_equal(x.ndim, 1)
    res = x[::-1] * x[1:, None]
    return np.ma.masked_where(res % 5 == 0, res)