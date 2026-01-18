import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_integer_split_2D_default(self):
    """ This will fail if we change default axis
        """
    a = np.array([np.arange(10), np.arange(10)])
    res = array_split(a, 3)
    tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]), np.zeros((0, 10))]
    compare_results(res, tgt)
    assert_(a.dtype.type is res[-1].dtype.type)