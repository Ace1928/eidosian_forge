import pytest
import cirq
import cirq.contrib.graph_device as ccgd
def test_empty_uniform_undirected_linear_device():
    n_qubits = 4
    edge_labels = {}
    device = ccgd.uniform_undirected_linear_device(n_qubits, edge_labels)
    assert device.qubits == tuple()
    assert device.edges == tuple()