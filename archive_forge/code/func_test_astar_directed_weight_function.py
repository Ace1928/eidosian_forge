import pytest
import networkx as nx
from networkx.utils import pairwise
def test_astar_directed_weight_function(self):
    w1 = lambda u, v, d: d['weight']
    assert nx.astar_path(self.XG, 'x', 'u', weight=w1) == ['x', 'u']
    assert nx.astar_path_length(self.XG, 'x', 'u', weight=w1) == 3
    assert nx.astar_path(self.XG, 's', 'v', weight=w1) == ['s', 'x', 'u', 'v']
    assert nx.astar_path_length(self.XG, 's', 'v', weight=w1) == 9
    w2 = lambda u, v, d: None if (u, v) == ('x', 'u') else d['weight']
    assert nx.astar_path(self.XG, 'x', 'u', weight=w2) == ['x', 'y', 's', 'u']
    assert nx.astar_path_length(self.XG, 'x', 'u', weight=w2) == 19
    assert nx.astar_path(self.XG, 's', 'v', weight=w2) == ['s', 'x', 'v']
    assert nx.astar_path_length(self.XG, 's', 'v', weight=w2) == 10
    w3 = lambda u, v, d: d['weight'] + 10
    assert nx.astar_path(self.XG, 'x', 'u', weight=w3) == ['x', 'u']
    assert nx.astar_path_length(self.XG, 'x', 'u', weight=w3) == 13
    assert nx.astar_path(self.XG, 's', 'v', weight=w3) == ['s', 'x', 'v']
    assert nx.astar_path_length(self.XG, 's', 'v', weight=w3) == 30