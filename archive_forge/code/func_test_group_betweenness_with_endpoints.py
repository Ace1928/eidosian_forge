import pytest
import networkx as nx
def test_group_betweenness_with_endpoints(self):
    """
        Group betweenness centrality for single node group
        """
    G = nx.path_graph(5)
    C = [1]
    b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False, endpoints=True)
    b_answer = 7.0
    assert b == b_answer