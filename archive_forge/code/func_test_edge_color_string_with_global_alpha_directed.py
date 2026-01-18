import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edge_color_string_with_global_alpha_directed():
    drawn_edges = nx.draw_networkx_edges(barbell.to_directed(), pos=nx.random_layout(barbell), edgelist=[(0, 1), (1, 2)], edge_color='purple', alpha=0.2)
    assert len(drawn_edges) == 2
    for fap in drawn_edges:
        ec = fap.get_edgecolor()
        assert mpl.colors.same_color(ec[:-1], 'purple')
        assert ec[-1] == 0.2