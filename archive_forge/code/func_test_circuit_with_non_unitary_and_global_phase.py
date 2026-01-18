import cirq
import pytest
def test_circuit_with_non_unitary_and_global_phase():
    device = cirq.testing.construct_ring_device(4)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([cirq.Moment(cirq.CNOT(q[0], q[1]), cirq.global_phase_operation(-1)), cirq.Moment(cirq.CNOT(q[1], q[2])), cirq.Moment(cirq.depolarize(0.1, 2).on(q[0], q[2]), cirq.depolarize(0.1).on(q[1]))])
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    expected = cirq.Circuit([cirq.Moment(cirq.CNOT(q[0], q[1]), cirq.global_phase_operation(-1)), cirq.Moment(cirq.CNOT(q[1], q[2])), cirq.Moment(cirq.depolarize(0.1).on(q[1])), cirq.Moment(cirq.SWAP(q[0], q[1])), cirq.Moment(cirq.depolarize(0.1, 2).on(q[1], q[2]))])
    cirq.testing.assert_same_circuits(routed_circuit, expected)