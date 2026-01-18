from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_chordless_cycles_directed(self):
    G = nx.DiGraph()
    nx.add_cycle(G, range(5))
    nx.add_cycle(G, range(4, 12))
    expected = [[*range(5)], [*range(4, 12)]]
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
    G.add_edge(7, 3)
    expected.append([*range(3, 8)])
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
    G.add_edge(3, 7)
    expected[-1] = [7, 3]
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)
    expected.pop()
    G.remove_edge(7, 3)
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 5], length_bound=5, chordless=True)