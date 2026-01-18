from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_gh6787_and_edge_attribute_names(self):
    G = nx.cycle_graph(4)
    G.add_weighted_edges_from([(0, 2, 10), (1, 3, 10)], weight='dist')
    expected = [[1, 3, 0], [3, 2, 1, 0], [1, 2, 0]]
    mcb = list(nx.minimum_cycle_basis(G, weight='dist'))
    assert len(mcb) == len(expected)
    assert all((c in expected for c in mcb))
    expected = [[1, 3, 0], [1, 2, 0], [3, 2, 0]]
    mcb = list(nx.minimum_cycle_basis(G))
    assert len(mcb) == len(expected)
    assert all((c in expected for c in mcb))