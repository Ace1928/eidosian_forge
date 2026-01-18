import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_colorsandweights(self):
    gm = self.GM(self.g1, self.g2, edge_match=self.emcm)
    assert not gm.is_isomorphic()