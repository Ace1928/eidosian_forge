import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edge_color_tuple_interpretation():
    """If edge_color is a sequence with the same length as edgelist, then each
    value in edge_color is mapped onto each edge via colormap."""
    G = nx.path_graph(6, create_using=nx.DiGraph)
    pos = {n: (n, n) for n in range(len(G))}
    for ec in ((0, 0, 1), (0, 0, 1, 1)):
        drawn_edges = nx.draw_networkx_edges(G, pos, edge_color=ec)
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)
        drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2)], edge_color=ec)
        for fap in drawn_edges:
            assert mpl.colors.same_color(fap.get_edgecolor(), ec)
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1, 1))
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1))
    for fap in drawn_edges:
        assert mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3)], edge_color=(0, 0, 1))
    assert mpl.colors.same_color(drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor())
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), 'blue')
    drawn_edges = nx.draw_networkx_edges(G, pos, edgelist=[(0, 1), (1, 2), (2, 3), (3, 4)], edge_color=(0, 0, 1, 1))
    assert mpl.colors.same_color(drawn_edges[0].get_edgecolor(), drawn_edges[1].get_edgecolor())
    assert mpl.colors.same_color(drawn_edges[2].get_edgecolor(), drawn_edges[3].get_edgecolor())
    for fap in drawn_edges:
        assert not mpl.colors.same_color(fap.get_edgecolor(), 'blue')