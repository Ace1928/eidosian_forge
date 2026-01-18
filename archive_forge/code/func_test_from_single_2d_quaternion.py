import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_single_2d_quaternion():
    x = np.array([[3, 4, 0, 0]])
    r = Rotation.from_quat(x)
    expected_quat = x / 5
    assert_array_almost_equal(r.as_quat(), expected_quat)