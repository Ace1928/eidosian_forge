import pytest
import networkx as nx
def test_floyd_warshall_predecessor_and_distance(self):
    XG = nx.DiGraph()
    XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
    path, dist = nx.floyd_warshall_predecessor_and_distance(XG)
    assert dist['s']['v'] == 9
    assert path['s']['v'] == 'u'
    assert dist == {'y': {'y': 0, 'x': 12, 's': 7, 'u': 15, 'v': 6}, 'x': {'y': 2, 'x': 0, 's': 9, 'u': 3, 'v': 4}, 's': {'y': 7, 'x': 5, 's': 0, 'u': 8, 'v': 9}, 'u': {'y': 2, 'x': 2, 's': 9, 'u': 0, 'v': 1}, 'v': {'y': 1, 'x': 13, 's': 8, 'u': 16, 'v': 0}}
    GG = XG.to_undirected()
    GG['u']['x']['weight'] = 2
    path, dist = nx.floyd_warshall_predecessor_and_distance(GG)
    assert dist['s']['v'] == 8
    G = nx.DiGraph()
    G.add_edges_from([('s', 'u'), ('s', 'x'), ('u', 'v'), ('u', 'x'), ('v', 'y'), ('x', 'u'), ('x', 'v'), ('x', 'y'), ('y', 's'), ('y', 'v')])
    path, dist = nx.floyd_warshall_predecessor_and_distance(G)
    assert dist['s']['v'] == 2
    dist = nx.floyd_warshall(G)
    assert dist['s']['v'] == 2
    XG = nx.DiGraph()
    XG.add_weighted_edges_from([('v', 'x', 5.0), ('y', 'x', 5.0), ('v', 'y', 6.0), ('x', 'u', 2.0)])
    path, dist = nx.floyd_warshall_predecessor_and_distance(XG)
    inf = float('inf')
    assert dist == {'v': {'v': 0, 'x': 5.0, 'y': 6.0, 'u': 7.0}, 'x': {'x': 0, 'u': 2.0, 'v': inf, 'y': inf}, 'y': {'y': 0, 'x': 5.0, 'v': inf, 'u': 7.0}, 'u': {'u': 0, 'v': inf, 'x': inf, 'y': inf}}
    assert path == {'v': {'x': 'v', 'y': 'v', 'u': 'x'}, 'x': {'u': 'x'}, 'y': {'x': 'y', 'u': 'x'}}