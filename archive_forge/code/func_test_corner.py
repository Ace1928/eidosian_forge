import pytest
from numpy import (
from numpy.testing import (
def test_corner(self):
    y = list(linspace(0, 1, 1))
    assert_(y == [0.0], y)
    assert_raises(TypeError, linspace, 0, 1, num=2.5)