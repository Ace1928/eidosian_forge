import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_graph_device_copy_and_add():
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    device_graph = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (c, d): None})
    crosstalk_graph = ccgd.UndirectedHypergraph(labelled_edges={(frozenset((a, b)), frozenset((c, d))): None})
    device = ccgd.UndirectedGraphDevice(device_graph=device_graph, crosstalk_graph=crosstalk_graph)
    device_graph_addend = ccgd.UndirectedHypergraph(labelled_edges={(a, b): None, (e, f): None})
    crosstalk_graph_addend = ccgd.UndirectedHypergraph(labelled_edges={(frozenset((a, b)), frozenset((e, f))): None})
    device_addend = ccgd.UndirectedGraphDevice(device_graph=device_graph_addend, crosstalk_graph=crosstalk_graph_addend)
    device_sum = device + device_addend
    device_copy = device.__copy__()
    device_copy += device_addend
    assert device != device_copy
    assert device_copy == device_sum