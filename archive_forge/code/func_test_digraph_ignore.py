from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_digraph_ignore(self):
    G = nx.DiGraph(self.edges)
    x = list(nx.find_cycle(G, self.nodes, orientation='ignore'))
    x_ = [(0, 1, FORWARD), (1, 0, FORWARD)]
    assert x == x_