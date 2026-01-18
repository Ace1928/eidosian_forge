import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_missing_target_node_paths():
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(4)
        list(nx.node_disjoint_paths(G, 1, 10))