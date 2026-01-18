import pyomo.common.unittest as unittest
from pyomo.common.dependencies import networkx as nx, networkx_available
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
def test_coarse_partition(self):
    bg = self._construct_graph()
    left_partition, right_partition = dulmage_mendelsohn(bg)
    left_underconstrained = {7, 8, 9, 10, 11}
    right_overconstrained = {21, 22}
    right_underconstrained = {12, 13, 14, 15, 16}
    left_overconstrained = {0, 1, 2}
    left_square = {3, 4, 5, 6}
    right_square = {17, 18, 19, 20}
    nodes = left_partition[0] + left_partition[1]
    self.assertEqual(set(nodes), left_underconstrained)
    nodes = right_partition[2]
    self.assertEqual(set(nodes), right_overconstrained)
    nodes = right_partition[0] + right_partition[1]
    self.assertEqual(set(nodes), right_underconstrained)
    nodes = left_partition[2]
    self.assertEqual(set(nodes), left_overconstrained)
    nodes = left_partition[3]
    self.assertEqual(set(nodes), left_square)
    nodes = right_partition[3]
    self.assertEqual(set(nodes), right_square)