import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_is_crosstalk_graph():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    assert not ccgd.is_crosstalk_graph('abc')
    graph = ccgd.UndirectedHypergraph()
    graph.add_vertex('abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), 'abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), None)
    graph.add_edge((frozenset((e, f)), frozenset((c, d))), lambda _: None)
    assert ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)), frozenset((c, d))), 'abc')
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((a, b)),), None)
    assert not ccgd.is_crosstalk_graph(graph)
    graph = ccgd.UndirectedHypergraph()
    graph.add_edge((frozenset((0, 1)), frozenset((2, 3))), None)
    assert not ccgd.is_crosstalk_graph(graph)