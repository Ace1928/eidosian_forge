import pytest
import networkx as nx
def test_non_negative_edge_weights():
    with pytest.raises(nx.NetworkXException):
        G = nx.path_graph(2)
        G.add_edge(0, 1, weight=-1)
        nx.second_order_centrality(G)