import pytest
import networkx as nx
def test_multipartite_layout_nonnumeric_partition_labels():
    """See gh-5123."""
    G = nx.Graph()
    G.add_node(0, subset='s0')
    G.add_node(1, subset='s0')
    G.add_node(2, subset='s1')
    G.add_node(3, subset='s1')
    G.add_edges_from([(0, 2), (0, 3), (1, 2)])
    pos = nx.multipartite_layout(G)
    assert len(pos) == len(G)