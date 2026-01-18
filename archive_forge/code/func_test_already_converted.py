import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_already_converted():
    q0 = cirq.LineQubit(0)
    c_orig = cirq.Circuit(cirq.PauliStringPhasor(cirq.X.on(q0)))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(single_qubit_target=CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS), ignore_failures=False)
    assert c_new == c_orig