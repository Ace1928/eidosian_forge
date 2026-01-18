import pytest
import networkx as nx
def test_group_betweenness_node_not_in_graph(self):
    """
        Node(s) in C not in graph, raises NodeNotFound exception
        """
    with pytest.raises(nx.NodeNotFound):
        nx.group_betweenness_centrality(nx.path_graph(5), [4, 7, 8])