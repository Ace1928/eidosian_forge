import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_same_circuits():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_same_circuits(cirq.Circuit(cirq.H(a)), cirq.Circuit(cirq.H(a)))
    with pytest.raises(AssertionError) as exc_info:
        cirq.testing.assert_same_circuits(cirq.Circuit(cirq.H(a)), cirq.Circuit())
    assert 'differing moment:\n0\n' in exc_info.value.args[0]
    with pytest.raises(AssertionError) as exc_info:
        cirq.testing.assert_same_circuits(cirq.Circuit(cirq.H(a), cirq.H(a)), cirq.Circuit(cirq.H(a), cirq.CZ(a, b)))
    assert 'differing moment:\n1\n' in exc_info.value.args[0]
    with pytest.raises(AssertionError):
        cirq.testing.assert_same_circuits(cirq.Circuit(cirq.CNOT(a, b)), cirq.Circuit(cirq.ControlledGate(cirq.X).on(a, b)))