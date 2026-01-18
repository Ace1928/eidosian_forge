import pytest
import networkx as nx
def test_bfs_layout_barbell():
    G = nx.barbell_graph(5, 3)
    pos = nx.bfs_layout(G, start=0)
    expected_nodes_per_layer = [1, 4, 1, 1, 1, 1, 4]
    assert np.array_equal(_num_nodes_per_bfs_layer(pos), expected_nodes_per_layer)
    pos = nx.bfs_layout(G, start=12)
    assert np.array_equal(_num_nodes_per_bfs_layer(pos), expected_nodes_per_layer)
    pos = nx.bfs_layout(G, start=6)
    expected_nodes_per_layer = [1, 2, 2, 8]
    assert np.array_equal(_num_nodes_per_bfs_layer(pos), expected_nodes_per_layer)