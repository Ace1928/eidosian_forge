import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_disconnected_graph():
    G = nx.fast_gnp_random_graph(100, 0.01, seed=42)
    cuts = nx.all_node_cuts(G)
    pytest.raises(nx.NetworkXError, next, cuts)