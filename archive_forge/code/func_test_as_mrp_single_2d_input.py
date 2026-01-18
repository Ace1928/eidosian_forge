import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_mrp_single_2d_input():
    quat = np.array([[1, 2, -3, 2]])
    expected_mrp = np.array([[0.16018862, 0.32037724, -0.48056586]])
    actual_mrp = Rotation.from_quat(quat).as_mrp()
    assert_equal(actual_mrp.shape, (1, 3))
    assert_allclose(actual_mrp, expected_mrp)