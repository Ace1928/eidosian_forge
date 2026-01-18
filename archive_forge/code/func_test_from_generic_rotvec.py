import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_generic_rotvec():
    rotvec = [[1, 2, 2], [1, -1, 0.5], [0, 0, 0]]
    expected_quat = np.array([[0.3324983, 0.6649967, 0.6649967, 0.0707372], [0.4544258, -0.4544258, 0.2272129, 0.7316889], [0, 0, 0, 1]])
    assert_array_almost_equal(Rotation.from_rotvec(rotvec).as_quat(), expected_quat)