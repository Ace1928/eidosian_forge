import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_digraph(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2)])
        self.func(G, 0, 2)