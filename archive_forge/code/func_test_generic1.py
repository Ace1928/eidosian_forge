import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_generic1(self):
    gm = self.GM(self.g1, self.g2, edge_match=self.emg1)
    assert gm.is_isomorphic()