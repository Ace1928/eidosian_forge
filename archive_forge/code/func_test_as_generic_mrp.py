import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_as_generic_mrp():
    quat = np.array([[1, 2, -1, 0.5], [1, -1, 1, 0.0003], [0, 0, 0, 1]])
    quat /= np.linalg.norm(quat, axis=1)[:, None]
    expected_mrp = np.array([[0.33333333, 0.66666667, -0.33333333], [0.57725028, -0.57725028, 0.57725028], [0, 0, 0]])
    assert_array_almost_equal(Rotation.from_quat(quat).as_mrp(), expected_mrp)