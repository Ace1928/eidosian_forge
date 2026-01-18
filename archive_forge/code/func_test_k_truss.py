import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_truss(self):
    k_truss_subgraph = nx.k_truss(self.G, -1)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 21))
    k_truss_subgraph = nx.k_truss(self.G, 0)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 21))
    k_truss_subgraph = nx.k_truss(self.G, 1)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 21))
    k_truss_subgraph = nx.k_truss(self.G, 2)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 21))
    k_truss_subgraph = nx.k_truss(self.G, 3)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 13))
    k_truss_subgraph = nx.k_truss(self.G, 4)
    assert sorted(k_truss_subgraph.nodes()) == list(range(1, 9))
    k_truss_subgraph = nx.k_truss(self.G, 5)
    assert sorted(k_truss_subgraph.nodes()) == []