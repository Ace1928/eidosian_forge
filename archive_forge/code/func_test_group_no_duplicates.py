import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('name, size', zip(NAMES, SIZES))
def test_group_no_duplicates(name, size):
    g = Rotation.create_group(name)
    kdtree = cKDTree(g.as_quat())
    assert len(kdtree.query_pairs(0.001)) == 0