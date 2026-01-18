import itertools
import os
import warnings
import pytest
import networkx as nx
def test_edgelist_kwarg_not_ignored():
    G = nx.path_graph(3)
    G.add_edge(0, 0)
    fig, ax = plt.subplots()
    nx.draw(G, edgelist=[(0, 1), (1, 2)], ax=ax)
    assert not ax.patches
    plt.delaxes(ax)