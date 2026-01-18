import cirq
from cirq.contrib.paulistring import pauli_string_optimized_circuit, CliffordTargetGateset
def test_handles_measurement_gate():
    q0, q1 = cirq.LineQubit.range(2)
    c_orig = cirq.Circuit(cirq.X(q0) ** 0.25, cirq.H(q0), cirq.CZ(q0, q1), cirq.H(q0), cirq.X(q0) ** 0.125, cirq.measure(q1, key='m1'), cirq.measure(q0, key='m0'))
    c_opt = pauli_string_optimized_circuit(c_orig)
    cirq.testing.assert_allclose_up_to_global_phase(c_orig.unitary(), c_opt.unitary(), atol=1e-07)
    cirq.testing.assert_has_diagram(c_opt, "\n0: ───[Y]^-0.5───@───[Z]^(-1/8)───[X]^0.5───[Z]^0.5───M('m0')───\n                 │\n1: ──────────────@───M('m1')────────────────────────────────────\n")