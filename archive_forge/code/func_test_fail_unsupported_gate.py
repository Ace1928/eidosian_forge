import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_fail_unsupported_gate():

    class UnsupportedGate(cirq.testing.TwoQubitGate):
        pass
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(UnsupportedGate()(q0, q1))
    with pytest.raises(ValueError):
        _ = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(), ignore_failures=False)