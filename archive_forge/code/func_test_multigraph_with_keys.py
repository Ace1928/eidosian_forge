import collections
import pytest
import networkx as nx
def test_multigraph_with_keys(self):
    G = nx.MultiGraph()
    nx.add_cycle(G, [0, 1, 2, 3])
    G.add_edge(1, 2)
    G.add_edge(1, 2)
    edges = list(nx.eulerian_circuit(G, source=0, keys=True))
    nodes = [u for u, v, k in edges]
    assert nodes == [0, 3, 2, 1, 2, 1]
    assert edges[:2] == [(0, 3, 0), (3, 2, 0)]
    assert collections.Counter(edges[2:5]) == collections.Counter([(2, 1, 0), (1, 2, 1), (2, 1, 2)])
    assert edges[5:] == [(1, 0, 0)]