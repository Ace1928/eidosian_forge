import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_is_separating_set_complete_graph():
    G = nx.complete_graph(5)
    assert _is_separating_set(G, {0, 1, 2, 3})