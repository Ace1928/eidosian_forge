import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
def test_colorsandweights2(self):
    self.g1.nodes['A']['color'] = 'blue'
    iso = nx.is_isomorphic(self.g1, self.g2, node_match=self.nm, edge_match=self.em)
    assert iso