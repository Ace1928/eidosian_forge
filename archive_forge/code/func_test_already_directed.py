import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_already_directed(self):
    uu = nx.to_undirected(self.uv)
    Muu = nx.to_undirected(self.Muv)
    assert edges_equal(uu.edges, self.uv.edges)
    assert edges_equal(Muu.edges, self.Muv.edges)