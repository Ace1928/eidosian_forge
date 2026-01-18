import pytest
import numpy as np
import sympy
import cirq
def test_assert_decompose_is_consistent_with_unitary():
    cirq.testing.assert_decompose_is_consistent_with_unitary(GoodGateDecompose())
    cirq.testing.assert_decompose_is_consistent_with_unitary(GoodGateDecompose().on(cirq.NamedQubit('q')))
    cirq.testing.assert_decompose_is_consistent_with_unitary(cirq.testing.PhaseUsingCleanAncilla(theta=0.1, ancilla_bitsize=3))
    cirq.testing.assert_decompose_is_consistent_with_unitary(cirq.testing.PhaseUsingDirtyAncilla(phase_state=1, ancilla_bitsize=4))
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_is_consistent_with_unitary(BadGateDecompose())
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_is_consistent_with_unitary(BadGateDecompose().on(cirq.NamedQubit('q')))