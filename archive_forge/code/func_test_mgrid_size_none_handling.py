import pytest
import numpy as np
from numpy.testing import (
from numpy.lib.index_tricks import (
@pytest.mark.parametrize('start, stop, step, expected', [(None, 10, 10j, (200, 10)), (-10, 20, None, (1800, 30))])
def test_mgrid_size_none_handling(self, start, stop, step, expected):
    grid = mgrid[start:stop:step, start:stop:step]
    grid_small = mgrid[start:stop:step]
    assert_equal(grid.size, expected[0])
    assert_equal(grid_small.size, expected[1])