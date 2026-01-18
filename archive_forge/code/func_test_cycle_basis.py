from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_cycle_basis(self):
    G = self.G
    cy = nx.cycle_basis(G, 0)
    sort_cy = sorted((sorted(c) for c in cy))
    assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
    cy = nx.cycle_basis(G, 1)
    sort_cy = sorted((sorted(c) for c in cy))
    assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
    cy = nx.cycle_basis(G, 9)
    sort_cy = sorted((sorted(c) for c in cy))
    assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5]]
    nx.add_cycle(G, 'ABC')
    cy = nx.cycle_basis(G, 9)
    sort_cy = sorted((sorted(c) for c in cy[:-1])) + [sorted(cy[-1])]
    assert sort_cy == [[0, 1, 2, 3], [0, 1, 6, 7, 8], [0, 3, 4, 5], ['A', 'B', 'C']]