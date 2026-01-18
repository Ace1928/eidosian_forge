import itertools
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_is_supported_operation():

    class MultiQubitOp(cirq.Operation):
        """Multi-qubit operation with unitary.

        Used to verify that `is_supported_operation` does not attempt to
        allocate the unitary for multi-qubit operations.
        """

        @property
        def qubits(self):
            return cirq.LineQubit.range(100)

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _has_unitary_(self):
            return True

        def _unitary_(self):
            assert False
    q1, q2 = cirq.LineQubit.range(2)
    assert cirq.CliffordSimulator.is_supported_operation(cirq.X(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.H(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.CNOT(q1, q2))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.measure(q1))
    assert cirq.CliffordSimulator.is_supported_operation(cirq.global_phase_operation(1j))
    assert not cirq.CliffordSimulator.is_supported_operation(cirq.T(q1))
    assert not cirq.CliffordSimulator.is_supported_operation(MultiQubitOp())