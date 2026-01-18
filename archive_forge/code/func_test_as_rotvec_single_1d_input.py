import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_rotvec_single_1d_input():
    quat = np.array([1, 2, -3, 2])
    expected_rotvec = np.array([0.5772381, 1.1544763, -1.7317144])
    actual_rotvec = Rotation.from_quat(quat).as_rotvec()
    assert_equal(actual_rotvec.shape, (3,))
    assert_allclose(actual_rotvec, expected_rotvec)