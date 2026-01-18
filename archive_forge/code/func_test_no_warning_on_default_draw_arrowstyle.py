import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize('draw_fn', (nx.draw, nx.draw_circular))
def test_no_warning_on_default_draw_arrowstyle(draw_fn):
    fig, ax = plt.subplots()
    G = nx.cycle_graph(5)
    with warnings.catch_warnings(record=True) as w:
        draw_fn(G, ax=ax)
    assert len(w) == 0
    plt.delaxes(ax)
    plt.close()