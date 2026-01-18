import pytest
import networkx as nx
def test_double_edge_swap():
    graph = nx.barabasi_albert_graph(200, 1)
    degrees = sorted((d for n, d in graph.degree()))
    G = nx.double_edge_swap(graph, 40)
    assert degrees == sorted((d for n, d in graph.degree()))