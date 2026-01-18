import cirq
import pytest
import numpy as np
import cirq_google as cg
import sympy
def assert_implements(circuit: cirq.Circuit, target_op: cirq.Operation):
    assert all((op in EXPECTED_TARGET_GATESET for op in circuit.all_operations()))
    assert sum((1 for _ in circuit.findall_operations(lambda e: len(e.qubits) > 2))) <= 6
    circuit.append(cirq.I.on_each(*target_op.qubits))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(circuit), cirq.unitary(target_op), atol=1e-07)