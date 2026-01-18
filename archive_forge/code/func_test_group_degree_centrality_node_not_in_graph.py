import pytest
import networkx as nx
def test_group_degree_centrality_node_not_in_graph(self):
    """
        Node(s) in S not in graph, raises NetworkXError
        """
    with pytest.raises(nx.NetworkXError):
        nx.group_degree_centrality(nx.path_graph(5), [6, 7, 8])