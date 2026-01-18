from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_cycle_basis2(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        G = nx.DiGraph()
        cy = nx.cycle_basis(G, 0)