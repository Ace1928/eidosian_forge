import pytest
import networkx as nx
def test_all_shortest_paths_zero_weight_edge(self):
    g = nx.Graph()
    nx.add_path(g, [0, 1, 3])
    nx.add_path(g, [0, 1, 2, 3])
    g.edges[1, 2]['weight'] = 0
    paths30d = list(nx.all_shortest_paths(g, 3, 0, weight='weight', method='dijkstra'))
    paths03d = list(nx.all_shortest_paths(g, 0, 3, weight='weight', method='dijkstra'))
    paths30b = list(nx.all_shortest_paths(g, 3, 0, weight='weight', method='bellman-ford'))
    paths03b = list(nx.all_shortest_paths(g, 0, 3, weight='weight', method='bellman-ford'))
    assert sorted(paths03d) == sorted((p[::-1] for p in paths30d))
    assert sorted(paths03d) == sorted((p[::-1] for p in paths30b))
    assert sorted(paths03b) == sorted((p[::-1] for p in paths30b))