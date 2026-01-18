import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
@pytest.mark.parametrize('n_qubits', (2, 5, 11))
def test_get_linear_device_graph(n_qubits):
    graph = ccr.get_linear_device_graph(n_qubits)
    assert sorted(graph) == cirq.LineQubit.range(n_qubits)
    assert len(graph.edges()) == n_qubits - 1
    assert all((abs(a.x - b.x) == 1 for a, b in graph.edges()))