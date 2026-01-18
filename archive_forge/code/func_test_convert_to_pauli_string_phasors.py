import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_convert_to_pauli_string_phasors():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.X(q0), cirq.Y(q1) ** 0.25, cirq.Z(q0) ** 0.125, cirq.H(q1))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(single_qubit_target=CliffordTargetGateset.SingleQubitTarget.PAULI_STRING_PHASORS))
    cirq.testing.assert_allclose_up_to_global_phase(c_new.unitary(), c_orig.unitary(), atol=1e-07)
    cirq.testing.assert_has_diagram(c_new, '\n0: ───[X]─────────[Z]^(1/8)───\n\n1: ───[Y]^-0.25───[Z]─────────\n')