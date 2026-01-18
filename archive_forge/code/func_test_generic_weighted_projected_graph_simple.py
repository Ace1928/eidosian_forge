import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_generic_weighted_projected_graph_simple(self):

    def shared(G, u, v):
        return len(set(G[u]) & set(G[v]))
    B = nx.path_graph(5)
    G = bipartite.generic_weighted_projected_graph(B, [0, 2, 4], weight_function=shared)
    assert nodes_equal(list(G), [0, 2, 4])
    assert edges_equal(list(G.edges(data=True)), [(0, 2, {'weight': 1}), (2, 4, {'weight': 1})])
    G = bipartite.generic_weighted_projected_graph(B, [0, 2, 4])
    assert nodes_equal(list(G), [0, 2, 4])
    assert edges_equal(list(G.edges(data=True)), [(0, 2, {'weight': 1}), (2, 4, {'weight': 1})])
    B = nx.DiGraph()
    nx.add_path(B, range(5))
    G = bipartite.generic_weighted_projected_graph(B, [0, 2, 4])
    assert nodes_equal(list(G), [0, 2, 4])
    assert edges_equal(list(G.edges(data=True)), [(0, 2, {'weight': 1}), (2, 4, {'weight': 1})])