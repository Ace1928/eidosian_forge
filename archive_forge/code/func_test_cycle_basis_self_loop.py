from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_cycle_basis_self_loop(self):
    """Tests the function for graphs with self loops"""
    G = nx.Graph()
    nx.add_cycle(G, [0, 1, 2, 3])
    nx.add_cycle(G, [0, 0, 6, 2])
    cy = nx.cycle_basis(G)
    sort_cy = sorted((sorted(c) for c in cy))
    assert sort_cy == [[0], [0, 1, 2], [0, 2, 3], [0, 2, 6]]