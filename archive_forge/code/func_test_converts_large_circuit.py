import pytest
import sympy
import cirq
from cirq.contrib.paulistring.clifford_target_gateset import CliffordTargetGateset
def test_converts_large_circuit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    before = cirq.Circuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q0), cirq.X(q0) ** 0.5, cirq.Y(q0) ** 0.5, cirq.Z(q0) ** 0.5, cirq.X(q0) ** (-0.5), cirq.Y(q0) ** (-0.5), cirq.Z(q0) ** (-0.5), cirq.H(q0), cirq.CZ(q0, q1), cirq.CZ(q1, q2), cirq.X(q0) ** 0.25, cirq.Y(q0) ** 0.25, cirq.Z(q0) ** 0.25, cirq.CZ(q0, q1))
    after = cirq.optimize_for_target_gateset(before, gateset=CliffordTargetGateset(), ignore_failures=False)
    cirq.testing.assert_allclose_up_to_global_phase(before.unitary(), after.unitary(), atol=1e-07)
    cirq.testing.assert_has_diagram(after, '\n0: ───Y^0.5───@───[Z]^-0.304───[X]^(1/3)───[Z]^0.446───────@───\n              │                                            │\n1: ───────────@────────────────────────────────────────@───@───\n                                                       │\n2: ────────────────────────────────────────────────────@───────\n')