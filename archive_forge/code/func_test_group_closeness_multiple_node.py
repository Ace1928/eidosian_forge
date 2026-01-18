import pytest
import networkx as nx
def test_group_closeness_multiple_node(self):
    """
        Group closeness centrality for a group with more than
        1 node
        """
    G = nx.path_graph(4)
    c = nx.group_closeness_centrality(G, [1, 2])
    c_answer = 1
    assert c == c_answer