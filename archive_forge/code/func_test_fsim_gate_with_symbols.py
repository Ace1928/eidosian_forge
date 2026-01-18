import itertools
import numpy as np
import pytest
import cirq
import sympy
def test_fsim_gate_with_symbols():
    theta, phi = sympy.symbols(['theta', 'phi'])
    op = cirq.FSimGate(theta=theta, phi=phi).on(*cirq.LineQubit.range(2))
    c_new_sqrt_iswap = cirq.Circuit(cirq.parameterized_2q_op_to_sqrt_iswap_operations(op))
    c_new_sqrt_iswap_inv = cirq.Circuit(cirq.parameterized_2q_op_to_sqrt_iswap_operations(op, use_sqrt_iswap_inv=True))
    for theta_val in np.linspace(0, 2 * np.pi, 4):
        for phi_val in np.linspace(0, 2 * np.pi, 6):
            cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.resolve_parameters(op, {'theta': theta_val, 'phi': phi_val})), cirq.unitary(cirq.resolve_parameters(c_new_sqrt_iswap, {'theta': theta_val, 'phi': phi_val})), atol=1e-06)
            cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.resolve_parameters(op, {'theta': theta_val, 'phi': phi_val})), cirq.unitary(cirq.resolve_parameters(c_new_sqrt_iswap_inv, {'theta': theta_val, 'phi': phi_val})), atol=1e-06)