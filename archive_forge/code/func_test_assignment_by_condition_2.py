from functools import reduce
import pytest
import numpy as np
import numpy.core.umath as umath
import numpy.core.fromnumeric as fromnumeric
from numpy.testing import (
from numpy.ma import (
from numpy.compat import pickle
def test_assignment_by_condition_2(self):
    a = masked_array([0, 1], mask=[False, False])
    b = masked_array([0, 1], mask=[True, True])
    mask = a < 1
    b[mask] = a[mask]
    expected_mask = [False, True]
    assert_equal(b.mask, expected_mask)