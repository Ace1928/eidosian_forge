import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_group_columns():
    structure = [[1, 1, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0], [0, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]
    for transform in [np.asarray, csr_matrix, csc_matrix, lil_matrix]:
        A = transform(structure)
        order = np.arange(6)
        groups_true = np.array([0, 1, 2, 0, 1, 2])
        groups = group_columns(A, order)
        assert_equal(groups, groups_true)
        order = [1, 2, 4, 3, 5, 0]
        groups_true = np.array([2, 0, 1, 2, 0, 1])
        groups = group_columns(A, order)
        assert_equal(groups, groups_true)
    groups_1 = group_columns(A)
    groups_2 = group_columns(A)
    assert_equal(groups_1, groups_2)