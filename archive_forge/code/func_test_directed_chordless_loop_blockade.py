from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_directed_chordless_loop_blockade(self):
    g = nx.DiGraph(((i, i) for i in range(10)))
    nx.add_cycle(g, range(10))
    expected_cycles = [(i,) for i in range(10)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)
    self.check_cycle_algorithm(g, expected_cycles, length_bound=1)
    g = nx.MultiDiGraph(g)
    g.add_edges_from(((i, i) for i in range(0, 10, 2)))
    expected_cycles = [(i,) for i in range(1, 10, 2)]
    self.check_cycle_algorithm(g, expected_cycles, chordless=True)