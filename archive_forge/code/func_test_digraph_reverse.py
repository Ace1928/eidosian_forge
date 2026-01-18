from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_reverse(self):
    G = nx.DiGraph(self.edges)
    x = list(nx.find_cycle(G, self.nodes, orientation='reverse'))
    x_ = [(1, 0, REVERSE), (0, 1, REVERSE)]
    assert x == x_