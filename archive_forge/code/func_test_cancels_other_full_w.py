from typing import cast, Iterable
import dataclasses
import numpy as np
import pytest
import sympy
import cirq
def test_cancels_other_full_w():
    q = cirq.NamedQubit('q')
    x = sympy.Symbol('x')
    y = sympy.Symbol('y')
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]), expected=quick_circuit())
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(q)], [cirq.PhasedXPowGate(phase_exponent=x).on(q)]), expected=quick_circuit(), eject_parameterized=True)
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.PhasedXPowGate(phase_exponent=0.125).on(q)]), expected=quick_circuit([cirq.Z(q) ** (-0.25)]))
    assert_optimizes(before=quick_circuit([cirq.X(q)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]), expected=quick_circuit([cirq.Z(q) ** 0.5]))
    assert_optimizes(before=quick_circuit([cirq.Y(q)], [cirq.PhasedXPowGate(phase_exponent=0.25).on(q)]), expected=quick_circuit([cirq.Z(q) ** (-0.5)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.X(q)]), expected=quick_circuit([cirq.Z(q) ** (-0.5)]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=0.25).on(q)], [cirq.Y(q)]), expected=quick_circuit([cirq.Z(q) ** 0.5]))
    assert_optimizes(before=quick_circuit([cirq.PhasedXPowGate(phase_exponent=x).on(q)], [cirq.PhasedXPowGate(phase_exponent=y).on(q)]), expected=quick_circuit([cirq.Z(q) ** (2 * (y - x))]), eject_parameterized=True)