import pytest
import networkx as nx
def test_bfs_layout_disconnected():
    G = nx.complete_graph(5)
    G.add_edges_from([(10, 11), (11, 12)])
    with pytest.raises(nx.NetworkXError, match="bfs_layout didn't include all nodes"):
        nx.bfs_layout(G, start=0)