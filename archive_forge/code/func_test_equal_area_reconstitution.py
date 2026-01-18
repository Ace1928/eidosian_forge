import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('poly', ['triangle', 'dodecagon', 'tetrahedron', 'cube', 'octahedron', 'dodecahedron', 'icosahedron'])
def test_equal_area_reconstitution(self, poly):
    points = _generate_polytope(poly)
    n, dim = points.shape
    sv = SphericalVoronoi(points)
    areas = sv.calculate_areas()
    assert_almost_equal(areas, _hypersphere_area(dim, 1) / n)