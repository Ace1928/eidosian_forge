import pytest
import networkx as nx
def test_all_pairs_shortest_path_length(self):
    ans = dict(nx.shortest_path_length(self.cycle))
    assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.all_pairs_shortest_path_length(self.cycle))
    ans = dict(nx.shortest_path_length(self.grid))
    assert ans[1][16] == 6
    ans = dict(nx.shortest_path_length(self.cycle, weight='weight'))
    assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.all_pairs_dijkstra_path_length(self.cycle))
    ans = dict(nx.shortest_path_length(self.grid, weight='weight'))
    assert ans[1][16] == 6
    ans = dict(nx.shortest_path_length(self.cycle, weight='weight', method='dijkstra'))
    assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.all_pairs_dijkstra_path_length(self.cycle))
    ans = dict(nx.shortest_path_length(self.cycle, weight='weight', method='bellman-ford'))
    assert ans[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
    assert ans == dict(nx.all_pairs_bellman_ford_path_length(self.cycle))