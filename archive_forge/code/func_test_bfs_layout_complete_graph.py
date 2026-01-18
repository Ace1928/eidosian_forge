import pytest
import networkx as nx
@pytest.mark.parametrize('n', range(2, 7))
def test_bfs_layout_complete_graph(n):
    """The complete graph should result in two layers: the starting node and
    a second layer containing all neighbors."""
    G = nx.complete_graph(n)
    pos = nx.bfs_layout(G, start=0)
    assert np.array_equal(_num_nodes_per_bfs_layer(pos), [1, n - 1])