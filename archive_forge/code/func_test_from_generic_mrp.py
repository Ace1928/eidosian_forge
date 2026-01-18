import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_generic_mrp():
    mrp = np.array([[1, 2, 2], [1, -1, 0.5], [0, 0, 0]])
    expected_quat = np.array([[0.2, 0.4, 0.4, -0.8], [0.61538462, -0.61538462, 0.30769231, -0.38461538], [0, 0, 0, 1]])
    assert_array_almost_equal(Rotation.from_mrp(mrp).as_quat(), expected_quat)