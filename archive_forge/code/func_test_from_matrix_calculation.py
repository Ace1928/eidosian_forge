import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_matrix_calculation():
    expected_quat = np.array([1, 1, 6, 1]) / np.sqrt(39)
    mat = np.array([[-0.8974359, -0.2564103, 0.3589744], [0.3589744, -0.8974359, 0.2564103], [0.2564103, 0.3589744, 0.8974359]])
    assert_array_almost_equal(Rotation.from_matrix(mat).as_quat(), expected_quat)
    assert_array_almost_equal(Rotation.from_matrix(mat.reshape((1, 3, 3))).as_quat(), expected_quat.reshape((1, 4)))