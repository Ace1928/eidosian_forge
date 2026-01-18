import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_apply_single_rotation_multiple_points():
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r1 = Rotation.from_matrix(mat)
    r2 = Rotation.from_matrix(np.expand_dims(mat, axis=0))
    v = np.array([[1, 2, 3], [4, 5, 6]])
    v_rotated = np.array([[-2, 1, 3], [-5, 4, 6]])
    assert_allclose(r1.apply(v), v_rotated)
    assert_allclose(r2.apply(v), v_rotated)
    v_inverse = np.array([[2, -1, 3], [5, -4, 6]])
    assert_allclose(r1.apply(v, inverse=True), v_inverse)
    assert_allclose(r2.apply(v, inverse=True), v_inverse)