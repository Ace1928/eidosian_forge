from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_dag(self):
    G = nx.DiGraph([(0, 1), (0, 2), (1, 2)])
    pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, orientation='original')
    x = list(nx.find_cycle(G, orientation='ignore'))
    assert x == [(0, 1, FORWARD), (1, 2, FORWARD), (0, 2, REVERSE)]