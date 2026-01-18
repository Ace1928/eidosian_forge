import cirq
import pytest
def test_circuit_with_multi_qubit_intermediate_measurement_gate_with_custom_key():
    device = cirq.testing.construct_ring_device(3)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    qs = cirq.LineQubit.range(3)
    hard_coded_mapper = cirq.HardCodedInitialMapper({qs[i]: qs[i] for i in range(3)})
    circuit = cirq.Circuit([cirq.Moment(cirq.measure(qs, key='test')), cirq.Moment(cirq.H.on_each(qs))])
    with pytest.raises(ValueError):
        _ = router(circuit, initial_mapper=hard_coded_mapper, context=cirq.TransformerContext(deep=True))