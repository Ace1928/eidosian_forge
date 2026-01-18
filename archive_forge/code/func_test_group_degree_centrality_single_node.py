import pytest
import networkx as nx
def test_group_degree_centrality_single_node(self):
    """
        Group degree centrality for a single node group
        """
    G = nx.path_graph(4)
    d = nx.group_degree_centrality(G, [1])
    d_answer = nx.degree_centrality(G)[1]
    assert d == d_answer