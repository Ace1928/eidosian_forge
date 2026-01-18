import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_1d_single_mrp():
    mrp = [0, 0, 1.0]
    expected_quat = np.array([0, 0, 1, 0])
    result = Rotation.from_mrp(mrp)
    assert_array_almost_equal(result.as_quat(), expected_quat)