from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.common.dependencies import scipy, scipy_available, networkx_available
import pyomo.common.unittest as unittest
def test_low_rank_hessenberg(self):
    """
        |x x      |
        |         |
        |      x  |
        |        x|
        |x x x x x|
        Know that first and last row and column will be in
        the imperfect matching.
        """
    N = 5
    omit = N // 2
    row = []
    col = []
    data = []
    for i in range(N):
        row.append(N - 1)
        col.append(i)
        data.append(1)
        if i == 0:
            row.append(0)
            col.append(i)
            data.append(1)
        elif i != omit:
            row.append(i - 1)
            col.append(i)
            data.append(1)
    matrix = sps.coo_matrix((data, (row, col)), shape=(N, N))
    matching = maximum_matching(matrix)
    values = set(matching.values())
    self.assertEqual(len(matching), N - 1)
    self.assertIn(0, matching)
    self.assertIn(N - 1, matching)
    self.assertIn(0, values)
    self.assertIn(N - 1, values)