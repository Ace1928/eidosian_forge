import pytest
import networkx as nx
from networkx.utils import nodes_equal
def test_k_truss_multigraph(self):
    G = nx.complete_graph(3)
    G = nx.MultiGraph(G)
    G.add_edge(1, 2)
    with pytest.raises(nx.NetworkXNotImplemented, match='not implemented for multigraph type'):
        nx.k_truss(G, k=1)