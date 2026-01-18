import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
from numpy.lib.stride_tricks import (
import pytest
def test_broadcast_shapes_succeeds():
    data = [[[], ()], [[()], ()], [[(7,)], (7,)], [[(1, 2), (2,)], (1, 2)], [[(1, 1)], (1, 1)], [[(1, 1), (3, 4)], (3, 4)], [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)], [[(5, 6, 1)], (5, 6, 1)], [[(1, 3), (3, 1)], (3, 3)], [[(1, 0), (0, 0)], (0, 0)], [[(0, 1), (0, 0)], (0, 0)], [[(1, 0), (0, 1)], (0, 0)], [[(1, 1), (0, 0)], (0, 0)], [[(1, 1), (1, 0)], (1, 0)], [[(1, 1), (0, 1)], (0, 1)], [[(), (0,)], (0,)], [[(0,), (0, 0)], (0, 0)], [[(0,), (0, 1)], (0, 0)], [[(1,), (0, 0)], (0, 0)], [[(), (0, 0)], (0, 0)], [[(1, 1), (0,)], (1, 0)], [[(1,), (0, 1)], (0, 1)], [[(1,), (1, 0)], (1, 0)], [[(), (1, 0)], (1, 0)], [[(), (0, 1)], (0, 1)], [[(1,), (3,)], (3,)], [[2, (3, 2)], (3, 2)]]
    for input_shapes, target_shape in data:
        assert_equal(broadcast_shapes(*input_shapes), target_shape)
    assert_equal(broadcast_shapes(*[(1, 2)] * 32), (1, 2))
    assert_equal(broadcast_shapes(*[(1, 2)] * 100), (1, 2))
    assert_equal(broadcast_shapes(*[(2,)] * 32), (2,))