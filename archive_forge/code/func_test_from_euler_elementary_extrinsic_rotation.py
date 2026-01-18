import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_euler_elementary_extrinsic_rotation():
    mat = Rotation.from_euler('zx', [90, 90], degrees=True).as_matrix()
    expected_mat = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    assert_array_almost_equal(mat, expected_mat)