import numpy as np
import itertools
from numpy.testing import (assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy.spatial import SphericalVoronoi, distance
from scipy.optimize import linear_sum_assignment
from scipy.constants import golden as phi
from scipy.special import gamma
@pytest.mark.parametrize('dim', range(2, 6))
def test_higher_dimensions(self, dim):
    n = 100
    points = _sample_sphere(n, dim, seed=0)
    sv = SphericalVoronoi(points)
    assert sv.vertices.shape[1] == dim
    assert len(sv.regions) == n
    cell_counts = []
    simplices = np.sort(sv._simplices)
    for i in range(1, dim + 1):
        cells = []
        for indices in itertools.combinations(range(dim), i):
            cells.append(simplices[:, list(indices)])
        cells = np.unique(np.concatenate(cells), axis=0)
        cell_counts.append(len(cells))
    expected_euler = 1 + (-1) ** (dim - 1)
    actual_euler = sum([(-1) ** i * e for i, e in enumerate(cell_counts)])
    assert expected_euler == actual_euler