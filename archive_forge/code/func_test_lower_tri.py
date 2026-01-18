import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
def test_lower_tri(self):
    """
        This matrix has a unique maximal matching and SCC
        order, making it a good test for a "fully decomposable"
        matrix.
        |x        |
        |x x      |
        |  x x    |
        |    x x  |
        |      x x|
        """
    N = 5
    row = []
    col = []
    data = []
    row.extend(range(N))
    col.extend(range(N))
    data.extend((1 for _ in range(N)))
    row.extend(range(1, N))
    col.extend(range(N - 1))
    data.extend((1 for _ in range(N - 1)))
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
    row_values = set(row_block_map.values())
    col_values = set(row_block_map.values())
    self.assertEqual(len(row_values), N)
    self.assertEqual(len(col_values), N)
    for i in range(N):
        self.assertEqual(row_block_map[i], i)
        self.assertEqual(col_block_map[i], i)