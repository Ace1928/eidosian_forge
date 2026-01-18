import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.utils import pairwise
def test_missing_source_edge_paths():
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(4)
        list(nx.edge_disjoint_paths(G, 10, 1))