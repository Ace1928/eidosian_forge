from itertools import product
import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest
from scipy.sparse import csr_matrix, coo_matrix, diags
from scipy.sparse.csgraph import (
@pytest.mark.parametrize('sign,test_case', linear_sum_assignment_test_cases)
def test_min_weight_full_matching_small_inputs(sign, test_case):
    linear_sum_assignment_assertions(min_weight_full_bipartite_matching, csr_matrix, sign, test_case)