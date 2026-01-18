import pytest
import networkx as nx
from networkx.utils import pairwise
def test_4_cycle(self):
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    dist, path = nx.single_source_bellman_ford(G, 0)
    assert dist == {0: 0, 1: 1, 2: 2, 3: 1}
    assert path[0] == [0]
    assert path[1] == [0, 1]
    assert path[2] in [[0, 1, 2], [0, 3, 2]]
    assert path[3] == [0, 3]
    pred, dist = nx.bellman_ford_predecessor_and_distance(G, 0)
    assert pred[0] == []
    assert pred[1] == [0]
    assert pred[2] in [[1, 3], [3, 1]]
    assert pred[3] == [0]
    assert dist == {0: 0, 1: 1, 2: 2, 3: 1}
    pred, dist = nx.goldberg_radzik(G, 0)
    assert pred[0] is None
    assert pred[1] == 0
    assert pred[2] in [1, 3]
    assert pred[3] == 0
    assert dist == {0: 0, 1: 1, 2: 2, 3: 1}