import pytest
import networkx as nx
from networkx.utils import pairwise
def test_dijkstra_predecessor2(self):
    G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
    pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
    assert pred[0] == []
    assert pred[1] == [0]
    assert pred[2] in [[1, 3], [3, 1]]
    assert pred[3] == [0]
    assert dist == {0: 0, 1: 1, 2: 2, 3: 1}