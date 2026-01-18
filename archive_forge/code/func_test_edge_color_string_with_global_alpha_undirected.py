import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edge_color_string_with_global_alpha_undirected():
    edge_collection = nx.draw_networkx_edges(barbell, pos=nx.random_layout(barbell), edgelist=[(0, 1), (1, 2)], edge_color='purple', alpha=0.2)
    ec = edge_collection.get_color().squeeze()
    assert len(edge_collection.get_paths()) == 2
    assert mpl.colors.same_color(ec[:-1], 'purple')
    assert ec[-1] == 0.2