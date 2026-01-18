from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_chordless_cycles_giant_hamiltonian(self):
    n = 1000
    assert n % 2 == 0
    G = nx.Graph()
    for v in range(n):
        if not v % 2:
            G.add_edge(v, (v + 2) % n)
        G.add_edge(v, (v + 1) % n)
    expected = [[*range(0, n, 2)]] + [[x % n for x in range(i, i + 3)] for i in range(0, n, 2)]
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 3], length_bound=3, chordless=True)
    n = 100
    assert n % 2 == 0
    G = nx.DiGraph()
    for v in range(n):
        G.add_edge(v, (v + 1) % n)
        if not v % 2:
            G.add_edge((v + 2) % n, v)
    expected = [[*range(n - 2, -2, -2)]] + [[x % n for x in range(i, i + 3)] for i in range(0, n, 2)]
    self.check_cycle_algorithm(G, expected, chordless=True)
    self.check_cycle_algorithm(G, [c for c in expected if len(c) <= 3], length_bound=3, chordless=True)