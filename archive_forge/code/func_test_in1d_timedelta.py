import numpy as np
from numpy.testing import (assert_array_equal, assert_equal,
from numpy.lib.arraysetops import (
import pytest
@pytest.mark.parametrize('kind', [None, 'sort'])
def test_in1d_timedelta(self, kind):
    """Test that in1d works for timedelta input"""
    rstate = np.random.RandomState(0)
    a = rstate.randint(0, 100, size=10)
    b = rstate.randint(0, 100, size=10)
    truth = in1d(a, b)
    a_timedelta = a.astype('timedelta64[s]')
    b_timedelta = b.astype('timedelta64[s]')
    assert_array_equal(truth, in1d(a_timedelta, b_timedelta, kind=kind))