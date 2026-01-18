import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_single_3d_matrix():
    mat = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).reshape((1, 3, 3))
    expected_quat = np.array([0.5, 0.5, 0.5, 0.5]).reshape((1, 4))
    assert_array_almost_equal(Rotation.from_matrix(mat).as_quat(), expected_quat)