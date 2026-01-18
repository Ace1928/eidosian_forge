import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_davenport_one_or_two_axes():
    ez = [0, 0, 1]
    ey = [0, 1, 0]
    rot = Rotation.from_rotvec(np.array(ez) * np.pi / 4)
    rot_dav = Rotation.from_davenport(ez, 'e', np.pi / 4)
    assert_allclose(rot.as_quat(canonical=True), rot_dav.as_quat(canonical=True))
    rot = Rotation.from_rotvec([np.array(ez) * np.pi / 4])
    rot_dav = Rotation.from_davenport([ez], 'e', [np.pi / 4])
    assert_allclose(rot.as_quat(canonical=True), rot_dav.as_quat(canonical=True))
    rot = Rotation.from_rotvec([np.array(ez) * np.pi / 4, np.array(ey) * np.pi / 6])
    rot = rot[0] * rot[1]
    rot_dav = Rotation.from_davenport([ey, ez], 'e', [np.pi / 6, np.pi / 4])
    assert_allclose(rot.as_quat(canonical=True), rot_dav.as_quat(canonical=True))
    rot = Rotation.from_rotvec([np.array(ez) * np.pi / 6, np.array(ez) * np.pi / 4])
    rot_dav = Rotation.from_davenport([ez], 'e', [np.pi / 6, np.pi / 4])
    assert_allclose(rot.as_quat(canonical=True), rot_dav.as_quat(canonical=True))