import pytest
from numpy import (
from numpy.testing import (
def test_nan_interior(self):
    with errstate(invalid='ignore'):
        y = geomspace(-3, 3, num=4)
    assert_equal(y[0], -3.0)
    assert_(isnan(y[1:-1]).all())
    assert_equal(y[3], 3.0)
    with errstate(invalid='ignore'):
        y = geomspace(-3, 3, num=4, endpoint=False)
    assert_equal(y[0], -3.0)
    assert_(isnan(y[1:]).all())