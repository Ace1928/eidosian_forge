from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
@pytest.mark.parametrize('num_rows,num_cols', [(0, 0), (2, 0), (0, 3)])
def test_linear_sum_assignment_trivial_cost(num_rows, num_cols):
    C = np.empty(shape=(num_cols, num_rows))
    row_ind, col_ind = linear_sum_assignment(C)
    assert len(row_ind) == 0
    assert len(col_ind) == 0