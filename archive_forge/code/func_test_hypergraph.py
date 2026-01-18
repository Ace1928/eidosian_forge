import random
import pytest
import cirq.contrib.graph_device as ccgd
def test_hypergraph():
    vertices = range(4)
    graph = ccgd.UndirectedHypergraph(vertices=vertices)
    assert graph.vertices == tuple(vertices)
    edges = [(0, 1), (2, 3)]
    graph = ccgd.UndirectedHypergraph(labelled_edges={edge: str(edge) for edge in edges})
    assert graph.vertices == tuple(vertices)
    graph.remove_vertex(0)
    assert graph.vertices == (1, 2, 3)
    assert graph.edges == (frozenset((2, 3)),)
    graph.remove_vertices((1, 3))
    assert graph.vertices == (2,)
    assert graph.edges == ()