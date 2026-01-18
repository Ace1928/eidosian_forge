from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_absorbs_z():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.625).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.PhasedXZGate(x_exponent=0, axis_phase_exponent=0, z_exponent=1).on(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.625).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXZGate(x_exponent=1, axis_phase_exponent=0.125, z_exponent=0).on(q)], [cirq.S(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.375).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q) ** x]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125 + x / 2).on(q)]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.Z(q) ** (x + 1)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.625 + x / 2).on(q)]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.S(q)], [cirq.T(q) ** (-1)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125).on(q)], [cirq.S(q) ** x], [cirq.T(q) ** (-x)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.125 + x * 0.125).on(q)]), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(q)], [cirq.S(q)]), expected=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x + 0.25).on(q)]), eject_parameterized=True)