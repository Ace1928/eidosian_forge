import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_convert_to_single_qubit_cliffords():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.X(q0), cirq.Y(q1) ** 0.5, cirq.Z(q0) ** (-0.5), cirq.Z(q1) ** 0, cirq.H(q0))
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=CliffordTargetGateset(single_qubit_target=CliffordTargetGateset.SingleQubitTarget.SINGLE_QUBIT_CLIFFORDS), ignore_failures=True)
    assert all((isinstance(op.gate, cirq.SingleQubitCliffordGate) for op in c_new.all_operations()))
    cirq.testing.assert_allclose_up_to_global_phase(c_new.unitary(), c_orig.unitary(), atol=1e-07)
    cirq.testing.assert_has_diagram(c_new, '\n0: ───(X^-0.5-Z^0.5)───\n\n1: ───Y^0.5────────────\n')