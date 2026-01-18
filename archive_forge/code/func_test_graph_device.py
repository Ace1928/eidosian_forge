import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def test_graph_device():
    one_qubit_duration = cirq.Duration(picos=10)
    two_qubit_duration = cirq.Duration(picos=1)
    one_qubit_edge = ccgd.FixedDurationUndirectedGraphDeviceEdge(one_qubit_duration)
    two_qubit_edge = ccgd.FixedDurationUndirectedGraphDeviceEdge(two_qubit_duration)
    empty_device = ccgd.UndirectedGraphDevice()
    assert not empty_device.qubits
    assert not empty_device.edges
    n_qubits = 4
    qubits = cirq.LineQubit.range(n_qubits)
    edges = {(cirq.LineQubit(i), cirq.LineQubit((i + 1) % n_qubits)): two_qubit_edge for i in range(n_qubits)}
    edges.update({(cirq.LineQubit(i),): one_qubit_edge for i in range(n_qubits)})
    device_graph = ccgd.UndirectedHypergraph(labelled_edges=edges)

    def not_cnots(first_op, second_op):
        if all((isinstance(op, cirq.GateOperation) and op.gate == cirq.CNOT for op in (first_op, second_op))):
            raise ValueError('Simultaneous CNOTs')
    assert ccgd.is_undirected_device_graph(device_graph)
    with pytest.raises(TypeError):
        ccgd.UndirectedGraphDevice('abc')
    constraint_edges = {(frozenset(cirq.LineQubit.range(2)), frozenset(cirq.LineQubit.range(2, 4))): None, (frozenset(cirq.LineQubit.range(1, 3)), frozenset((cirq.LineQubit(0), cirq.LineQubit(3)))): not_cnots}
    crosstalk_graph = ccgd.UndirectedHypergraph(labelled_edges=constraint_edges)
    assert ccgd.is_crosstalk_graph(crosstalk_graph)
    with pytest.raises(TypeError):
        ccgd.UndirectedGraphDevice(device_graph, crosstalk_graph='abc')
    graph_device = ccgd.UndirectedGraphDevice(device_graph)
    assert graph_device.crosstalk_graph == ccgd.UndirectedHypergraph()
    graph_device = ccgd.UndirectedGraphDevice(device_graph, crosstalk_graph=crosstalk_graph)
    assert sorted(graph_device.edges) == sorted(device_graph.edges)
    assert graph_device.qubits == tuple(qubits)
    assert graph_device.device_graph == device_graph
    assert graph_device.labelled_edges == device_graph.labelled_edges
    assert graph_device.duration_of(cirq.X(qubits[2])) == one_qubit_duration
    assert graph_device.duration_of(cirq.CNOT(*qubits[:2])) == two_qubit_duration
    with pytest.raises(KeyError):
        graph_device.duration_of(cirq.CNOT(qubits[0], qubits[2]))
    with pytest.raises(ValueError):
        graph_device.validate_operation(cirq.CNOT(qubits[0], qubits[2]))
    with pytest.raises(AttributeError):
        graph_device.validate_operation(list((2, 3)))
    moment = cirq.Moment([cirq.CNOT(*qubits[:2]), cirq.CNOT(*qubits[2:])])
    with pytest.raises(ValueError):
        graph_device.validate_moment(moment)
    moment = cirq.Moment([cirq.CNOT(qubits[0], qubits[3]), cirq.CZ(qubits[1], qubits[2])])
    graph_device.validate_moment(moment)
    moment = cirq.Moment([cirq.CNOT(qubits[0], qubits[3]), cirq.CNOT(qubits[1], qubits[2])])
    with pytest.raises(ValueError):
        graph_device.validate_moment(moment)