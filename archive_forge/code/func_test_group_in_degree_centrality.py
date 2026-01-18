import pytest
import networkx as nx
def test_group_in_degree_centrality(self):
    """
        Group in-degree centrality in a DiGraph
        """
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8])
    G.add_edges_from([(1, 2), (1, 3), (1, 6), (1, 7), (1, 8), (2, 3), (2, 4), (2, 5)])
    d = nx.group_in_degree_centrality(G, [1, 2])
    d_answer = 0
    assert d == d_answer