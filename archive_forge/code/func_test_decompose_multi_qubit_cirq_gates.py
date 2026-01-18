import cirq
import cirq_ionq as ionq
import pytest
import sympy
@pytest.mark.parametrize('gate, qubits', [(cirq.CCZ, 3), (cirq.QuantumFourierTransformGate(6), 6), (cirq.MatrixGate(cirq.testing.random_unitary(8)), 3)])
def test_decompose_multi_qubit_cirq_gates(gate, qubits):
    circuit = cirq.Circuit(gate(*cirq.LineQubit.range(qubits)))
    decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ionq_target_gateset, ignore_failures=False)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, decomposed_circuit, atol=1e-08)
    assert ionq_target_gateset.validate(decomposed_circuit)