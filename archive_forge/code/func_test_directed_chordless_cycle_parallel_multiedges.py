from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_directed_chordless_cycle_parallel_multiedges(self):
    g = nx.MultiGraph()
    nx.add_cycle(g, range(5))
    expected = [[*range(5)]]
    self.check_cycle_algorithm(g, expected, chordless=True)
    nx.add_cycle(g, range(5))
    expected = [*cycle_edges(range(5))]
    self.check_cycle_algorithm(g, expected, chordless=True)
    nx.add_cycle(g, range(5))
    expected = []
    self.check_cycle_algorithm(g, expected, chordless=True)
    g = nx.MultiDiGraph()
    nx.add_cycle(g, range(5))
    expected = [[*range(5)]]
    self.check_cycle_algorithm(g, expected, chordless=True)
    nx.add_cycle(g, range(5))
    self.check_cycle_algorithm(g, [], chordless=True)
    nx.add_cycle(g, range(5))
    self.check_cycle_algorithm(g, [], chordless=True)
    g = nx.MultiDiGraph()
    nx.add_cycle(g, range(5))
    nx.add_cycle(g, range(5)[::-1])
    expected = [*cycle_edges(range(5))]
    self.check_cycle_algorithm(g, expected, chordless=True)
    nx.add_cycle(g, range(5))
    self.check_cycle_algorithm(g, [], chordless=True)