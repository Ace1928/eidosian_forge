import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_generic_weighted_projected_graph_custom(self):

    def jaccard(G, u, v):
        unbrs = set(G[u])
        vnbrs = set(G[v])
        return len(unbrs & vnbrs) / len(unbrs | vnbrs)

    def my_weight(G, u, v, weight='weight'):
        w = 0
        for nbr in set(G[u]) & set(G[v]):
            w += G.edges[u, nbr].get(weight, 1) + G.edges[v, nbr].get(weight, 1)
        return w
    B = nx.bipartite.complete_bipartite_graph(2, 2)
    for i, (u, v) in enumerate(B.edges()):
        B.edges[u, v]['weight'] = i + 1
    G = bipartite.generic_weighted_projected_graph(B, [0, 1], weight_function=jaccard)
    assert edges_equal(list(G.edges(data=True)), [(0, 1, {'weight': 1.0})])
    G = bipartite.generic_weighted_projected_graph(B, [0, 1], weight_function=my_weight)
    assert edges_equal(list(G.edges(data=True)), [(0, 1, {'weight': 10})])
    G = bipartite.generic_weighted_projected_graph(B, [0, 1])
    assert edges_equal(list(G.edges(data=True)), [(0, 1, {'weight': 2})])