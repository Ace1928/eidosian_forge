import pytest
import networkx as nx
def test_directed_edge_swap_tries():
    with pytest.raises(nx.NetworkXError):
        G = nx.directed_edge_swap(nx.path_graph(3, create_using=nx.DiGraph), nswap=1, max_tries=0)