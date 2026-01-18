import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_align_vectors_near_inf():
    n = 100
    mats = []
    for i in range(6):
        mats.append(Rotation.random(n, random_state=10 + i).as_matrix())
    for i in range(n):
        a = [1 * mats[0][i][0], 2 * mats[1][i][0]]
        b = [3 * mats[2][i][0], 4 * mats[3][i][0]]
        R, _ = Rotation.align_vectors(a, b, weights=[10000000000.0, 1])
        R2, _ = Rotation.align_vectors(a, b, weights=[np.inf, 1])
        assert_allclose(R.as_matrix(), R2.as_matrix(), atol=0.0001)
    for i in range(n):
        a = [1 * mats[0][i][0], 2 * mats[1][i][0], 3 * mats[2][i][0]]
        b = [4 * mats[3][i][0], 5 * mats[4][i][0], 6 * mats[5][i][0]]
        R, _ = Rotation.align_vectors(a, b, weights=[10000000000.0, 2, 1])
        R2, _ = Rotation.align_vectors(a, b, weights=[np.inf, 2, 1])
        assert_allclose(R.as_matrix(), R2.as_matrix(), atol=0.0001)