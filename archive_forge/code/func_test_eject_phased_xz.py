from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_eject_phased_xz():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit(cirq.PhasedXZGate(x_exponent=1, z_exponent=0.5, axis_phase_exponent=0.5).on(a), cirq.CZ(a, b) ** 0.25)
    c_expected = cirq.Circuit(cirq.CZ(a, b) ** (-0.25), cirq.PhasedXPowGate(phase_exponent=0.75).on(a), cirq.T(b))
    cirq.testing.assert_same_circuits(cirq.eject_z(cirq.eject_phased_paulis(cirq.eject_z(c))), c_expected)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c, c_expected, 1e-08)