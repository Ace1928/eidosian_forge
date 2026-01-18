from typing import Optional, Sequence, Type
import pytest
import cirq
import sympy
import numpy as np
def test_avoids_decompose_when_matrix_available():

    class OtherXX(cirq.testing.TwoQubitGate):

        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False

    class OtherOtherXX(cirq.testing.TwoQubitGate):

        def _has_unitary_(self) -> bool:
            return True

        def _unitary_(self) -> np.ndarray:
            m = np.array([[0, 1], [1, 0]])
            return np.kron(m, m)

        def _decompose_(self, qubits):
            assert False
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(OtherXX()(a, b), OtherOtherXX()(a, b))
    c = cirq.optimize_for_target_gateset(c, gateset=cirq.CZTargetGateset(), ignore_failures=False)
    assert len(c) == 0