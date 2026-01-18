import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_euler_intrinsic_rotation_313():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    mat = Rotation.from_euler('ZXZ', angles, degrees=True).as_matrix()
    assert_array_almost_equal(mat[0], np.array([[0.43559574, -0.78914913, 0.4330127], [0.65973961, -0.04736717, -0.75], [0.61237244, 0.61237244, 0.5]]))
    assert_array_almost_equal(mat[1], np.array([[0.625, -0.64951905, 0.4330127], [0.64951905, 0.125, -0.75], [0.4330127, 0.75, 0.5]]))
    assert_array_almost_equal(mat[2], np.array([[-0.1767767, -0.91855865, 0.35355339], [0.88388348, -0.30618622, -0.35355339], [0.4330127, 0.25, 0.8660254]]))