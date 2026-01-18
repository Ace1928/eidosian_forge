import cirq
import cirq_ionq as ionq
import pytest
import sympy
def test_decompose_parameterized_operation():
    op = cirq.ISWAP(*cirq.LineQubit.range(2))
    theta = sympy.Symbol('theta')
    circuit = cirq.Circuit(op ** theta)
    decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ionq_target_gateset, ignore_failures=False)
    for theta_val in [-0.25, 1.0, 0.5]:
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(cirq.resolve_parameters(circuit, {theta: theta_val}), cirq.resolve_parameters(decomposed_circuit, {theta: theta_val}), atol=1e-06)
    assert ionq_target_gateset.validate(decomposed_circuit)