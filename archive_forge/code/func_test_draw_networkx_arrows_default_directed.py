import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('drawing_func', (nx.draw, nx.draw_networkx))
def test_draw_networkx_arrows_default_directed(drawing_func):
    import matplotlib.collections
    G = nx.path_graph(3, create_using=nx.DiGraph)
    fig, ax = plt.subplots()
    drawing_func(G, ax=ax)
    assert not any((isinstance(c, mpl.collections.LineCollection) for c in ax.collections))
    assert ax.patches
    plt.delaxes(ax)