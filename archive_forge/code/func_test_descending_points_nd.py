import itertools
import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_warns,
from pytest import raises as assert_raises
from scipy.interpolate import (RegularGridInterpolator, interpn,
from scipy.sparse._sputils import matrix
from scipy._lib._util import ComplexWarning
@parametrize_rgi_interp_methods
@pytest.mark.parametrize(('ndims', 'func'), [(2, lambda x, y: 2 * x ** 3 + 3 * y ** 2), (3, lambda x, y, z: 2 * x ** 3 + 3 * y ** 2 - z), (4, lambda x, y, z, a: 2 * x ** 3 + 3 * y ** 2 - z + a), (5, lambda x, y, z, a, b: 2 * x ** 3 + 3 * y ** 2 - z + a * b)])
def test_descending_points_nd(self, method, ndims, func):
    rng = np.random.default_rng(42)
    sample_low = 1
    sample_high = 5
    test_points = rng.uniform(sample_low, sample_high, size=(2, ndims))
    ascending_points = [np.linspace(sample_low, sample_high, 12) for _ in range(ndims)]
    ascending_values = func(*np.meshgrid(*ascending_points, indexing='ij', sparse=True))
    ascending_interp = RegularGridInterpolator(ascending_points, ascending_values, method=method)
    ascending_result = ascending_interp(test_points)
    descending_points = [xi[::-1] for xi in ascending_points]
    descending_values = func(*np.meshgrid(*descending_points, indexing='ij', sparse=True))
    descending_interp = RegularGridInterpolator(descending_points, descending_values, method=method)
    descending_result = descending_interp(test_points)
    assert_array_equal(ascending_result, descending_result)