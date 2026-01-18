import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_euler_intrinsic_rotation_312():
    angles = [[30, 60, 45], [30, 60, 30], [45, 30, 60]]
    mat = Rotation.from_euler('ZXY', angles, degrees=True).as_matrix()
    assert_array_almost_equal(mat[0], np.array([[0.3061862, -0.25, 0.9185587], [0.8838835, 0.4330127, -0.1767767], [-0.3535534, 0.8660254, 0.3535534]]))
    assert_array_almost_equal(mat[1], np.array([[0.5334936, -0.25, 0.8080127], [0.8080127, 0.4330127, -0.3995191], [-0.25, 0.8660254, 0.4330127]]))
    assert_array_almost_equal(mat[2], np.array([[0.0473672, -0.6123725, 0.7891491], [0.6597396, 0.6123725, 0.4355958], [-0.75, 0.5, 0.4330127]]))