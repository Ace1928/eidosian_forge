import numpy as np
from numpy.testing import assert_equal, assert_array_equal
import pytest
from scipy.stats import rankdata, tiecorrect
from scipy._lib._util import np_long
def test_no_correction(self):
    """Arrays with no ties require no correction."""
    ranks = np.arange(2.0)
    c = tiecorrect(ranks)
    assert_equal(c, 1.0)
    ranks = np.arange(3.0)
    c = tiecorrect(ranks)
    assert_equal(c, 1.0)