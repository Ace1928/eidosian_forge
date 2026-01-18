import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_colors_only(self):
    gm = self.GM(self.g1, self.g2, edge_match=self.emc)
    assert gm.is_isomorphic()