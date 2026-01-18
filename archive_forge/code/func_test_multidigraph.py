from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_multidigraph(self):
    with pytest.raises(nx.NetworkXNotImplemented):
        nx.dag_to_branching(nx.MultiDiGraph())