import pytest
import networkx as nx
def test_directed_edge_swap_small():
    with pytest.raises(nx.NetworkXError):
        G = nx.directed_edge_swap(nx.path_graph(3, create_using=nx.DiGraph))