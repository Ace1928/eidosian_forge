import pytest
import networkx as nx
def test_fewer_than_4_nodes():
    G = nx.DiGraph()
    G.add_nodes_from([0, 1, 2])
    with pytest.raises(nx.NetworkXError, match='.*fewer than four nodes.'):
        nx.directed_edge_swap(G)