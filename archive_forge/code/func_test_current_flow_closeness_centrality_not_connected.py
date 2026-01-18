import pytest
import networkx as nx
def test_current_flow_closeness_centrality_not_connected(self):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3])
    with pytest.raises(nx.NetworkXError):
        nx.current_flow_closeness_centrality(G)