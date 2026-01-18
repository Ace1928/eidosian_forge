import random
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.common.dependencies import networkx_available
from pyomo.common.dependencies import scipy_available
from pyomo.common.collections import ComponentSet, ComponentMap
from pyomo.contrib.incidence_analysis.interface import (
from pyomo.contrib.incidence_analysis.connected import get_independent_submatrices
from pyomo.contrib.incidence_analysis.tests.models_for_testing import (
import pyomo.common.unittest as unittest
def test_decomposable_matrix(self):
    """
        The following matrix decomposes into two independent diagonal
        blocks.
        | x         |
        | x x       |
        |     x x   |
        |       x x |
        |         x |
        """
    row = [0, 1, 1, 2, 2, 3, 3, 4]
    col = [0, 0, 1, 2, 3, 3, 4, 4]
    data = [1, 1, 1, 1, 1, 1, 1, 1]
    N = 5
    coo = sp.sparse.coo_matrix((data, (row, col)), shape=(N, N))
    row_blocks, col_blocks = get_independent_submatrices(coo)
    self.assertEqual(len(row_blocks), 2)
    self.assertEqual(len(col_blocks), 2)
    self.assertTrue(set(row_blocks[0]) == {0, 1} or set(row_blocks[1]) == {0, 1})
    self.assertTrue(set(col_blocks[0]) == {0, 1} or set(col_blocks[1]) == {0, 1})
    self.assertTrue(set(row_blocks[0]) == {2, 3, 4} or set(row_blocks[1]) == {2, 3, 4})
    self.assertTrue(set(col_blocks[0]) == {2, 3, 4} or set(col_blocks[1]) == {2, 3, 4})