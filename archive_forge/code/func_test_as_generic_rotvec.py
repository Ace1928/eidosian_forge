import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_generic_rotvec():
    quat = np.array([[1, 2, -1, 0.5], [1, -1, 1, 0.0003], [0, 0, 0, 1]])
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    rotvec = Rotation.from_quat(quat).as_rotvec()
    angle = np.linalg.norm(rotvec, axis=1)
    assert_allclose(quat[:, 3], np.cos(angle / 2))
    assert_allclose(np.cross(rotvec, quat[:, :3]), np.zeros((3, 3)))