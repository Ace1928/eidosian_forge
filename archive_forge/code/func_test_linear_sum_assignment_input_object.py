from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
def test_linear_sum_assignment_input_object():
    C = [[1, 2, 3], [4, 5, 6]]
    assert_array_equal(linear_sum_assignment(C), linear_sum_assignment(np.asarray(C)))
    assert_array_equal(linear_sum_assignment(C), linear_sum_assignment(matrix(C)))