import pytest
import networkx as nx
from networkx.generators.classic import empty_graph
from networkx.utils import edges_equal, nodes_equal
def test_convert_node_labels_to_integers(self):
    G = empty_graph()
    H = nx.convert_node_labels_to_integers(G, 100)
    assert list(H.nodes()) == []
    assert list(H.edges()) == []
    for opt in ['default', 'sorted', 'increasing degree', 'decreasing degree']:
        G = empty_graph()
        H = nx.convert_node_labels_to_integers(G, 100, ordering=opt)
        assert list(H.nodes()) == []
        assert list(H.edges()) == []
    G = empty_graph()
    G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'D')])
    H = nx.convert_node_labels_to_integers(G)
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    H = nx.convert_node_labels_to_integers(G, 1000)
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    assert nodes_equal(H.nodes(), [1000, 1001, 1002, 1003])
    H = nx.convert_node_labels_to_integers(G, ordering='increasing degree')
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    assert H.degree(0) == 1
    assert H.degree(1) == 2
    assert H.degree(2) == 2
    assert H.degree(3) == 3
    H = nx.convert_node_labels_to_integers(G, ordering='decreasing degree')
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    assert H.degree(0) == 3
    assert H.degree(1) == 2
    assert H.degree(2) == 2
    assert H.degree(3) == 1
    H = nx.convert_node_labels_to_integers(G, ordering='increasing degree', label_attribute='label')
    degH = (d for n, d in H.degree())
    degG = (d for n, d in G.degree())
    assert sorted(degH) == sorted(degG)
    assert H.degree(0) == 1
    assert H.degree(1) == 2
    assert H.degree(2) == 2
    assert H.degree(3) == 3
    assert H.nodes[3]['label'] == 'C'
    assert H.nodes[0]['label'] == 'D'
    assert H.nodes[1]['label'] == 'A' or H.nodes[2]['label'] == 'A'
    assert H.nodes[1]['label'] == 'B' or H.nodes[2]['label'] == 'B'