import cirq
import pytest
def test_circuit_with_two_qubit_intermediate_measurement_gate():
    device = cirq.testing.construct_ring_device(2)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    qs = cirq.LineQubit.range(2)
    hard_coded_mapper = cirq.HardCodedInitialMapper({qs[i]: qs[i] for i in range(2)})
    circuit = cirq.Circuit([cirq.Moment(cirq.measure(qs)), cirq.Moment(cirq.H.on_each(qs))])
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper, context=cirq.TransformerContext(deep=True))
    device.validate_circuit(routed_circuit)