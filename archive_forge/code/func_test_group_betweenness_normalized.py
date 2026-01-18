import pytest
import networkx as nx
def test_group_betweenness_normalized(self):
    """
        Group betweenness centrality for group with more than
        1 node and normalized
        """
    G = nx.path_graph(5)
    C = [1, 3]
    b = nx.group_betweenness_centrality(G, C, weight=None, normalized=True, endpoints=False)
    b_answer = 1.0
    assert b == b_answer