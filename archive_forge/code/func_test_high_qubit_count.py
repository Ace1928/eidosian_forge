import cirq
import pytest
def test_high_qubit_count():
    c_orig = cirq.testing.random_circuit(qubits=40, n_moments=350, op_density=0.4, random_state=0)
    device = cirq.testing.construct_grid_device(7, 7)
    device_graph = device.metadata.nx_graph
    router = cirq.RouteCQC(device_graph)
    c_routed = router(c_orig)
    device.validate_circuit(c_routed)