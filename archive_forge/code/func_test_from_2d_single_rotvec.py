import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_2d_single_rotvec():
    rotvec = [[1, 0, 0]]
    expected_quat = np.array([[0.4794255, 0, 0, 0.8775826]])
    result = Rotation.from_rotvec(rotvec)
    assert_array_almost_equal(result.as_quat(), expected_quat)