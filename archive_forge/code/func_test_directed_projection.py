import pytest
import networkx as nx
from networkx.algorithms import bipartite
from networkx.utils import edges_equal, nodes_equal
def test_directed_projection(self):
    G = nx.DiGraph()
    G.add_edge('A', 1)
    G.add_edge(1, 'B')
    G.add_edge('A', 2)
    G.add_edge('B', 2)
    P = bipartite.projected_graph(G, 'AB')
    assert edges_equal(list(P.edges()), [('A', 'B')])
    P = bipartite.weighted_projected_graph(G, 'AB')
    assert edges_equal(list(P.edges()), [('A', 'B')])
    assert P['A']['B']['weight'] == 1
    P = bipartite.projected_graph(G, 'AB', multigraph=True)
    assert edges_equal(list(P.edges()), [('A', 'B')])
    G = nx.DiGraph()
    G.add_edge('A', 1)
    G.add_edge(1, 'B')
    G.add_edge('A', 2)
    G.add_edge(2, 'B')
    P = bipartite.projected_graph(G, 'AB')
    assert edges_equal(list(P.edges()), [('A', 'B')])
    P = bipartite.weighted_projected_graph(G, 'AB')
    assert edges_equal(list(P.edges()), [('A', 'B')])
    assert P['A']['B']['weight'] == 2
    P = bipartite.projected_graph(G, 'AB', multigraph=True)
    assert edges_equal(list(P.edges()), [('A', 'B'), ('A', 'B')])