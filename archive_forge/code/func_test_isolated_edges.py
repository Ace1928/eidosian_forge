import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_isolated_edges():
    with pytest.raises(nx.NetworkXNoPath):
        G = nx.Graph()
        G.add_node(1)
        nx.add_path(G, [4, 5])
        list(nx.edge_disjoint_paths(G, 1, 5))