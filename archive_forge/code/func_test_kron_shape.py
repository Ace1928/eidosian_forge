import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
@pytest.mark.parametrize('shape_a,shape_b', [((1, 1), (1, 1)), ((1, 2, 3), (4, 5, 6)), ((2, 2), (2, 2, 2)), ((1, 0), (1, 1)), ((2, 0, 2), (2, 2)), ((2, 0, 0, 2), (2, 0, 2))])
def test_kron_shape(self, shape_a, shape_b):
    a = np.ones(shape_a)
    b = np.ones(shape_b)
    normalised_shape_a = (1,) * max(0, len(shape_b) - len(shape_a)) + shape_a
    normalised_shape_b = (1,) * max(0, len(shape_a) - len(shape_b)) + shape_b
    expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)
    k = np.kron(a, b)
    assert np.array_equal(k.shape, expected_shape), 'Unexpected shape from kron'