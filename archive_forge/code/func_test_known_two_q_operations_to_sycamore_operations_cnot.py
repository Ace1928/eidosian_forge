import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_known_two_q_operations_to_sycamore_operations_cnot():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(a, b))
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)
    multi_qubit_ops = [e for e in converted_circuit.all_operations() if len(e.qubits) > 1]
    assert len(multi_qubit_ops) == 2
    assert all((isinstance(e.gate, cirq_google.SycamoreGate) for e in multi_qubit_ops))