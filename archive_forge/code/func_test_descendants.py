from collections import deque
from itertools import combinations, permutations
import pytest
import networkx as nx
from networkx.utils import edges_equal, pairwise
def test_descendants(self):
    G = nx.DiGraph()
    descendants = nx.algorithms.dag.descendants
    G.add_edges_from([(1, 2), (1, 3), (4, 2), (4, 3), (4, 5), (2, 6), (5, 6)])
    assert descendants(G, 1) == {2, 3, 6}
    assert descendants(G, 4) == {2, 3, 5, 6}
    assert descendants(G, 3) == set()
    pytest.raises(nx.NetworkXError, descendants, G, 8)