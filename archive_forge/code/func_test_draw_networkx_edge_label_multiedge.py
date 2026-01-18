import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_networkx_edge_label_multiedge():
    G = nx.MultiGraph()
    G.add_edge(0, 1, weight=10)
    G.add_edge(0, 1, weight=20)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    pos = {n: (n, n) for n in G}
    text_items = nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, connectionstyle=['arc3,rad=0.1', 'arc3,rad=0.2'])
    assert len(text_items) == 2