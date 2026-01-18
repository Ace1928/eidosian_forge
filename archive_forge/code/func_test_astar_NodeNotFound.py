import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_NodeNotFound(self):
    """Tests that exception is raised when either
        source or target is not in graph"""
    G = nx.gnp_random_graph(10, 0.2, seed=10)
    with pytest.raises(nx.NodeNotFound):
        nx.astar_path_length(G, 11, 9)