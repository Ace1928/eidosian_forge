import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_networkx_edge_label_multiedge_exception():
    """
    draw_networkx_edge_labels should raise an informative error message when
    the edge label includes keys
    """
    exception_msg = 'draw_networkx_edge_labels does not support multiedges'
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 1, weight=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    pos = {n: (n, n) for n in G}
    with pytest.raises(nx.NetworkXError, match=exception_msg):
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)