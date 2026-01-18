import random
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.triangularize import (
from pyomo.common.dependencies import (
import pyomo.common.unittest as unittest
def test_low_rank_exception(self):
    N = 5
    row = list(range(N - 1))
    col = list(range(N - 1))
    data = [1 for _ in range(N - 1)]
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    with self.assertRaises(RuntimeError) as exc:
        row_block_map, col_block_map = block_triangularize(matrix)
    self.assertIn('perfect matching', str(exc.exception))