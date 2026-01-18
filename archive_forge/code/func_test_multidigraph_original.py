from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_multidigraph_original(self):
    G = nx.MultiDiGraph([(0, 1), (1, 2), (2, 3), (4, 2)])
    pytest.raises(nx.exception.NetworkXNoCycle, nx.find_cycle, G, [0, 1, 2, 3, 4], orientation='original')