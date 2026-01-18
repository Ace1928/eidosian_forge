from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_graph_orientation_none(self):
    G = nx.Graph(self.edges)
    G.add_edge(2, 0)
    x = list(nx.find_cycle(G, self.nodes, orientation=None))
    x_ = [(0, 1), (1, 2), (2, 0)]
    assert x == x_