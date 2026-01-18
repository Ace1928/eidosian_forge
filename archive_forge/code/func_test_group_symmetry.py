import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('name, size', zip(NAMES, SIZES))
def test_group_symmetry(name, size):
    g = Rotation.create_group(name)
    q = np.concatenate((-g.as_quat(), g.as_quat()))
    distance = np.sort(cdist(q, q))
    deltas = np.max(distance, axis=0) - np.min(distance, axis=0)
    assert (deltas < TOL).all()