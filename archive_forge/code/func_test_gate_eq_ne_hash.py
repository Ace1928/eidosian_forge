import itertools
import pytest
import numpy as np
import sympy
import cirq
def test_gate_eq_ne_hash():
    eq = cirq.testing.EqualsTester()
    dps_xyx = cirq.DensePauliString('XYX')
    eq.make_equality_group(lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=0.5), lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=-1.5), lambda: cirq.PauliStringPhasorGate(dps_empty, exponent_neg=2.5))
    eq.make_equality_group(lambda: cirq.PauliStringPhasorGate(-dps_empty, exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyz), cirq.PauliStringPhasorGate(dps_xyz, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyz, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyx), cirq.PauliStringPhasorGate(dps_xyx, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xy), cirq.PauliStringPhasorGate(dps_xy, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_yx), cirq.PauliStringPhasorGate(dps_yx, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyx, exponent_neg=1))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyx, exponent_neg=0.5))
    eq.add_equality_group(cirq.PauliStringPhasorGate(-dps_xyx, exponent_neg=-0.5))
    eq.add_equality_group(cirq.PauliStringPhasorGate(dps_xyz, exponent_neg=sympy.Symbol('a')))