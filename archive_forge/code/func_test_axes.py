import itertools
import os
import warnings
import pytest
import networkx as nx
def test_axes():
    fig, ax = plt.subplots()
    nx.draw(barbell, ax=ax)
    nx.draw_networkx_edge_labels(barbell, nx.circular_layout(barbell), ax=ax)