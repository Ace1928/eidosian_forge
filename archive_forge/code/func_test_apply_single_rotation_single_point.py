import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_apply_single_rotation_single_point():
    mat = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    r_1d = Rotation.from_matrix(mat)
    r_2d = Rotation.from_matrix(np.expand_dims(mat, axis=0))
    v_1d = np.array([1, 2, 3])
    v_2d = np.expand_dims(v_1d, axis=0)
    v1d_rotated = np.array([-2, 1, 3])
    v2d_rotated = np.expand_dims(v1d_rotated, axis=0)
    assert_allclose(r_1d.apply(v_1d), v1d_rotated)
    assert_allclose(r_1d.apply(v_2d), v2d_rotated)
    assert_allclose(r_2d.apply(v_1d), v2d_rotated)
    assert_allclose(r_2d.apply(v_2d), v2d_rotated)
    v1d_inverse = np.array([2, -1, 3])
    v2d_inverse = np.expand_dims(v1d_inverse, axis=0)
    assert_allclose(r_1d.apply(v_1d, inverse=True), v1d_inverse)
    assert_allclose(r_1d.apply(v_2d, inverse=True), v2d_inverse)
    assert_allclose(r_2d.apply(v_1d, inverse=True), v2d_inverse)
    assert_allclose(r_2d.apply(v_2d, inverse=True), v2d_inverse)