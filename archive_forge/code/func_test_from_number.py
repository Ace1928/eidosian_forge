import numpy as np
from numpy.testing import assert_equal
from pytest import raises as assert_raises
from scipy.io._harwell_boeing import (
def test_from_number(self):
    f = np.array([1.0, -1.2])
    r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]
    for i, j in zip(f, r_f):
        assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)