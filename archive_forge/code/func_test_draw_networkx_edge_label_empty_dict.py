import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_networkx_edge_label_empty_dict():
    """Regression test for draw_networkx_edge_labels with empty dict. See
    gh-5372."""
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G.nodes}
    assert nx.draw_networkx_edge_labels(G, pos, edge_labels={}) == {}