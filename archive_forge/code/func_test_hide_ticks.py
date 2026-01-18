import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('hide_ticks', [False, True])
@pytest.mark.parametrize('method', [nx.draw_networkx, nx.draw_networkx_edge_labels, nx.draw_networkx_edges, nx.draw_networkx_labels, nx.draw_networkx_nodes])
def test_hide_ticks(method, hide_ticks):
    G = nx.path_graph(3)
    pos = {n: (n, n) for n in G.nodes}
    _, ax = plt.subplots()
    method(G, pos=pos, ax=ax, hide_ticks=hide_ticks)
    for axis in [ax.xaxis, ax.yaxis]:
        assert bool(axis.get_ticklabels()) != hide_ticks
    plt.delaxes(ax)
    plt.close()