import pytest
import networkx as nx
def test_all_pairs_shortest_path(self):
    p = nx.shortest_path(self.cycle)
    assert p[0][3] == [0, 1, 2, 3]
    assert p == dict(nx.all_pairs_shortest_path(self.cycle))
    p = nx.shortest_path(self.grid)
    validate_grid_path(4, 4, 1, 12, p[1][12])
    p = nx.shortest_path(self.cycle, weight='weight')
    assert p[0][3] == [0, 1, 2, 3]
    assert p == dict(nx.all_pairs_dijkstra_path(self.cycle))
    p = nx.shortest_path(self.grid, weight='weight')
    validate_grid_path(4, 4, 1, 12, p[1][12])
    p = nx.shortest_path(self.cycle, weight='weight', method='dijkstra')
    assert p[0][3] == [0, 1, 2, 3]
    assert p == dict(nx.all_pairs_dijkstra_path(self.cycle))
    p = nx.shortest_path(self.cycle, weight='weight', method='bellman-ford')
    assert p[0][3] == [0, 1, 2, 3]
    assert p == dict(nx.all_pairs_bellman_ford_path(self.cycle))