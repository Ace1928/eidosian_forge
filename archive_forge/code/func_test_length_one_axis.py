import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
def test_length_one_axis(self):
    values = np.array([[0.1, 1, 10]])
    xi = np.array([[1, 2.2], [1, 3.2], [1, 3.8]])
    res = interpn(([1], [2, 3, 4]), values, xi)
    wanted = [0.9 * 0.2 + 0.1, 9 * 0.2 + 1, 9 * 0.8 + 1]
    assert_allclose(res, wanted, atol=1e-15)
    xi = np.array([[1.1, 2.2], [1.5, 3.2], [-2.3, 3.8]])
    res = interpn(([1], [2, 3, 4]), values, xi, bounds_error=False, fill_value=None)
    assert_allclose(res, wanted, atol=1e-15)