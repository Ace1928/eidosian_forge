import cirq
import cirq_ionq as ionq
import pytest
import sympy
def test_decompose_two_qubit_matrix_gate():
    q0, q1 = cirq.LineQubit.range(2)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(4))
        circuit = cirq.Circuit(gate(q0, q1))
        decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ionq_target_gateset, ignore_failures=False)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, decomposed_circuit, atol=1e-08)
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)