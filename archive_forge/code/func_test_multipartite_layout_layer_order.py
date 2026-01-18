import pytest
import networkx as nx
def test_multipartite_layout_layer_order():
    """Return the layers in sorted order if the layers of the multipartite
    graph are sortable. See gh-5691"""
    G = nx.Graph()
    for node, layer in zip(('a', 'b', 'c', 'd', 'e'), (2, 3, 1, 2, 4)):
        G.add_node(node, subset=layer)
    pos = nx.multipartite_layout(G, align='horizontal')
    assert pos['a'][-1] == pos['d'][-1]
    assert pos['c'][-1] < pos['a'][-1] < pos['b'][-1] < pos['e'][-1]
    G.nodes['a']['subset'] = 'layer_0'
    pos_nosort = nx.multipartite_layout(G)
    assert pos_nosort.keys() == pos.keys()