from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_is_aperiodic_disconnected2():
    G = nx.DiGraph()
    nx.add_cycle(G, [0, 1, 2])
    G.add_edge(3, 3)
    assert not nx.is_aperiodic(G)