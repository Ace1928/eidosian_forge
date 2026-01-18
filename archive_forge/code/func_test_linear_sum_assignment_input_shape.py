from numpy.testing import assert_array_equal
import pytest
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import random
from scipy.sparse._sputils import matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.sparse.csgraph.tests.test_matching import (
def test_linear_sum_assignment_input_shape():
    with pytest.raises(ValueError, match='expected a matrix'):
        linear_sum_assignment([1, 2, 3])