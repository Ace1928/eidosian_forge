from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_multidigraph_ignore2(self):
    G = nx.MultiDiGraph([(0, 1), (1, 2), (1, 2)])
    x = list(nx.find_cycle(G, [0, 1, 2], orientation='ignore'))
    x_ = [(1, 2, 0, FORWARD), (1, 2, 1, REVERSE)]
    assert x == x_