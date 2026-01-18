import numpy as np
import scipy.stats
import cirq
def test_decompose_x():
    """Verifies correctness of multi-controlled X decomposition."""
    for total_qubits_count in range(1, 8):
        qubits = cirq.LineQubit.range(total_qubits_count)
        for controls_count in range(total_qubits_count):
            gates = cirq.decompose_multi_controlled_x(qubits[:controls_count], qubits[controls_count], qubits[controls_count + 1:])
            circuit1 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit1.append(gates)
            result_matrix = circuit1.unitary()
            circuit2 = cirq.Circuit([cirq.I.on(q) for q in qubits])
            circuit2 += cirq.ControlledGate(cirq.X, num_controls=controls_count).on(*qubits[0:controls_count + 1])
            expected_matrix = circuit2.unitary()
            assert np.allclose(expected_matrix, result_matrix)