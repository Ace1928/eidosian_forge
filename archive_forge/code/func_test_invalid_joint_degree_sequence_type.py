import pytest
import networkx as nx
def test_invalid_joint_degree_sequence_type(self):
    with pytest.raises(nx.NetworkXError, match='Invalid degree sequence'):
        nx.random_clustered_graph([[1, 1], [2, 1], [0, 1]])