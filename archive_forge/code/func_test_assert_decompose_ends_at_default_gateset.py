import pytest
import numpy as np
import sympy
import cirq
def test_assert_decompose_ends_at_default_gateset():
    cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposesToDefaultGateset())
    cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposesToDefaultGateset().on(*cirq.LineQubit.range(2)))
    cirq.testing.assert_decompose_ends_at_default_gateset(ParameterizedGate())
    cirq.testing.assert_decompose_ends_at_default_gateset(ParameterizedGate().on(*cirq.LineQubit.range(2)))
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposeNotImplemented())
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposeNotImplemented().on(cirq.NamedQubit('q')))
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposeDoesNotEndInDefaultGateset())
    with pytest.raises(AssertionError):
        cirq.testing.assert_decompose_ends_at_default_gateset(GateDecomposeDoesNotEndInDefaultGateset().on(*cirq.LineQubit.range(4)))