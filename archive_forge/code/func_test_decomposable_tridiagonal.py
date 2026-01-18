import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
def test_decomposable_tridiagonal(self):
    """
        This matrix decomposes into 2x2 blocks
        |x x      |
        |x x      |
        |  x x x  |
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
    row.extend((i for i in range(N - 1) if not i % 2))
    col.extend((i + 1 for i in range(N - 1) if not i % 2))
    data.extend((1 for i in range(N - 1) if not i % 2))
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    row_block_map, col_block_map = map_coords_to_block_triangular_indices(matrix)
    row_values = set(row_block_map.values())
    col_values = set(row_block_map.values())
    self.assertEqual(len(row_values), (N + 1) // 2)
    self.assertEqual(len(col_values), (N + 1) // 2)
    for i in range((N + 1) // 2):
        self.assertEqual(row_block_map[2 * i], i)
        self.assertEqual(col_block_map[2 * i], i)
        if 2 * i + 1 < N:
            self.assertEqual(row_block_map[2 * i + 1], i)
            self.assertEqual(col_block_map[2 * i + 1], i)