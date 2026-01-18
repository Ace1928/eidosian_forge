import itertools
import os
import warnings
import pytest
import networkx as nx
def test_draw_networkx_edges_undirected_selfloop_colors():
    """When an edgelist is supplied along with a sequence of colors, check that
    the self-loops have the correct colors."""
    fig, ax = plt.subplots()
    edgelist = [(1, 3), (1, 2), (2, 3), (1, 1), (3, 3), (2, 2)]
    edge_colors = ['pink', 'cyan', 'black', 'red', 'blue', 'green']
    G = nx.Graph(edgelist)
    pos = {n: (n, n) for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edgelist, edge_color=edge_colors)
    assert len(ax.patches) == 3
    sl_points = np.array(edgelist[-3:]) + np.array([0, 0.1])
    for fap, clr, slp in zip(ax.patches, edge_colors[-3:], sl_points):
        assert fap.get_path().contains_point(slp)
        assert mpl.colors.same_color(fap.get_edgecolor(), clr)
    plt.delaxes(ax)