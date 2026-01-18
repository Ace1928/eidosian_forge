import pytest
import networkx as nx
def test_non_connected():
    with pytest.raises(nx.NetworkXException):
        G = nx.Graph()
        G.add_node(0)
        G.add_node(1)
        nx.second_order_centrality(G)