import pytest
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import special_ortho_group
from itertools import permutations
import pickle
import copy
def test_from_matrix_ortho_output():
    rnd = np.random.RandomState(0)
    mat = rnd.random_sample((100, 3, 3))
    ortho_mat = Rotation.from_matrix(mat).as_matrix()
    mult_result = np.einsum('...ij,...jk->...ik', ortho_mat, ortho_mat.transpose((0, 2, 1)))
    eye3d = np.zeros((100, 3, 3))
    for i in range(3):
        eye3d[:, i, i] = 1.0
    assert_array_almost_equal(mult_result, eye3d)