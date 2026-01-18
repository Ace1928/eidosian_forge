import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('n', [10, 500])
@pytest.mark.parametrize('dim', [2, 3])
@pytest.mark.parametrize('radius', [0.5, 1, 2])
@pytest.mark.parametrize('shift', [False, True])
@pytest.mark.parametrize('single_hemisphere', [False, True])
def test_area_reconstitution(self, n, dim, radius, shift, single_hemisphere):
    points = _sample_sphere(n, dim, seed=0)
    if single_hemisphere:
        points[:, 0] = np.abs(points[:, 0])
    center = (np.arange(dim) + 1) * shift
    points = radius * points + center
    sv = SphericalVoronoi(points, radius=radius, center=center)
    areas = sv.calculate_areas()
    assert_almost_equal(areas.sum(), _hypersphere_area(dim, radius))