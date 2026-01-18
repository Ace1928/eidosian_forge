import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_full_graph(self):
    G = self.K3
    H = nx.induced_subgraph(G, [0, 1, 2, 5])
    assert H.name == G.name
    self.graphs_equal(H, G)
    self.same_attrdict(H, G)