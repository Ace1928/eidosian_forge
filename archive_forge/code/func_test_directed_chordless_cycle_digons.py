from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_directed_chordless_cycle_digons(self):
    g = nx.DiGraph()
    nx.add_cycle(g, range(5))
    nx.add_cycle(g, range(5)[::-1])
    g.add_edge(0, 0)
    expected_cycles = [(0,), (1, 2), (2, 3), (3, 4)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=2)
    expected_cycles = [c for c in expected_cycles if len(c) < 2]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True, length_bound=1)