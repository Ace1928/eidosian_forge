import warnings
import itertools
import pytest
import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.ma.extras import (
def test_masked_all_like(self):
    base = array([1, 2], dtype=float)
    test = masked_all_like(base)
    control = array([1, 1], mask=[1, 1], dtype=float)
    assert_equal(test, control)
    dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
    base = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
    test = masked_all_like(base)
    control = array([(10, 10), (10, 10)], mask=[(1, 1), (1, 1)], dtype=dt)
    assert_equal(test, control)
    dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
    control = array([(1, (1, 1)), (1, (1, 1))], mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
    test = masked_all_like(control)
    assert_equal(test, control)