import pytest
import networkx as nx
def test_custom_joint_degree_sequence(self):
    node = [1, 1, 1, 2, 1, 2, 0, 0]
    tri = [0, 0, 0, 0, 0, 1, 1, 1]
    joint_degree_sequence = zip(node, tri)
    G = nx.random_clustered_graph(joint_degree_sequence)
    assert G.number_of_nodes() == 8
    assert G.number_of_edges() == 7