import pytest
import networkx as nx
from networkx.algorithms import node_classification
def test_labeled_nodes_are_not_changed(self):
    G = nx.karate_club_graph()
    label_name = 'club'
    label_removed = {0, 1, 2, 3, 4, 5, 6, 7}
    for i in label_removed:
        del G.nodes[i][label_name]
    predicted = node_classification.harmonic_function(G, label_name=label_name)
    label_not_removed = set(range(len(G))) - label_removed
    for i in label_not_removed:
        assert predicted[i] == G.nodes[i][label_name]