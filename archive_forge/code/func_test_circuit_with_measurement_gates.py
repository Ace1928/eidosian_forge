import cirq
import pytest
def test_circuit_with_measurement_gates():
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.MeasurementGate(2).on(q[0], q[2]), cirq.MeasurementGate(3).on(*q))
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    cirq.testing.assert_same_circuits(routed_circuit, circuit)