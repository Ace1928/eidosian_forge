import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_n_rotations():
    mat = np.empty((2, 3, 3))
    mat[0] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    mat[1] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    r = Rotation.from_matrix(mat)
    assert_equal(len(r), 2)
    assert_equal(len(r[:-1]), 1)