import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
def test_area_unsupported_dimension(self):
    dim = 4
    points = np.concatenate((-np.eye(dim), np.eye(dim)))
    sv = SphericalVoronoi(points)
    with pytest.raises(TypeError, match='Only supported'):
        sv.calculate_areas()