import cirq
import pytest
def test_empty_circuit():
    device = cirq.testing.construct_grid_device(5, 5)
    device_graph = device.metadata.nx_graph
    empty_circuit = cirq.Circuit()
    router = cirq.RouteCQC(device_graph)
    empty_circuit_routed = router(empty_circuit)
    device.validate_circuit(empty_circuit_routed)
    assert len(list(empty_circuit.all_operations())) == len(list(empty_circuit_routed.all_operations()))