import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.constants import golden as phi
from scipy.spatial import cKDTree
@pytest.mark.parametrize('axis', ['A', 'b', 0, 1, 2, 4, False, None])
def test_axis_valid(axis):
    with pytest.raises(ValueError, match='`axis` must be one of'):
        Rotation.create_group('C1', axis)