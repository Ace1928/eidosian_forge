import cirq
import pytest
def test_already_valid_circuit():
    device = cirq.testing.construct_ring_device(10)
    device_graph = device.metadata.nx_graph
    circuit = cirq.Circuit([cirq.Moment(cirq.CNOT(cirq.LineQubit(i), cirq.LineQubit(i + 1))) for i in range(9)], cirq.X(cirq.LineQubit(1)))
    hard_coded_mapper = cirq.HardCodedInitialMapper({cirq.LineQubit(i): cirq.LineQubit(i) for i in range(10)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    device.validate_circuit(routed_circuit)
    cirq.testing.assert_same_circuits(routed_circuit, circuit)