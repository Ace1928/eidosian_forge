import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
def test_decomposable_bordered(self):
    """
        This matrix decomposes
        |x        |
        |  x      |
        |    x   x|
        |      x x|
        |x x x x  |
        """
    N = 5
    half = N // 2
    row = []
    col = []
    data = []
    row.extend(range(N - 1))
    col.extend(range(N - 1))
    data.extend((1 for _ in range(N - 1)))
    row.extend((N - 1 for _ in range(N - 1)))
    col.extend(range(N - 1))
    data.extend((1 for _ in range(N - 1)))
    row.extend(range(half, N - 1))
    col.extend((N - 1 for _ in range(half, N - 1)))
    data.extend((1 for _ in range(half, N - 1)))
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
    row_values = set(row_block_map.values())
    col_values = set(row_block_map.values())
    self.assertEqual(len(row_values), half + 1)
    self.assertEqual(len(col_values), half + 1)
    first_half_set = set(range(half))
    for i in range(N):
        if i < half:
            self.assertIn(row_block_map[i], first_half_set)
            self.assertIn(col_block_map[i], first_half_set)
        else:
            self.assertEqual(row_block_map[i], half)
            self.assertEqual(col_block_map[i], half)