import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_convert_to_integers2(self):
    G = empty_graph()
    G.add_edges_from([('C', 'D'), ('A', 'B'), ('A', 'C'), ('B', 'C')])
    H = nx.convert_node_labels_to_integers(G, ordering='sorted')
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    H = nx.convert_node_labels_to_integers(G, ordering='sorted', label_attribute='label')
    assert H.nodes[0]['label'] == 'A'
    assert H.nodes[1]['label'] == 'B'
    assert H.nodes[2]['label'] == 'C'
    assert H.nodes[3]['label'] == 'D'