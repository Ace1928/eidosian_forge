import pytest
from numpy import (
from numpy.testing import (
@pytest.mark.parametrize('axis', [0, 1, -1])
def test_base_array(self, axis: int):
    start = 1
    stop = 2
    num = 6
    base = array([1, 2])
    t1 = logspace(start, stop, num=num, base=base, axis=axis)
    t2 = stack([logspace(start, stop, num=num, base=_base) for _base in base], axis=(axis + 1) % t1.ndim)
    assert_equal(t1, t2)