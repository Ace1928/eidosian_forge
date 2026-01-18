import random
import numpy as np
import pytest
import sympy
import cirq
def test_canonicalization():

    def f(x, z, a):
        return cirq.PhasedXZGate(x_exponent=x, z_exponent=z, axis_phase_exponent=a)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(f(-1, 0, 0), f(-3, 0, 0), f(1, 1, 0.5))
    '\n    # Canonicalize X exponent (-1, +1].\n    if isinstance(x, numbers.Real):\n        x %= 2\n        if x > 1:\n            x -= 2\n    # Axis phase exponent is irrelevant if there is no X exponent.\n    # Canonicalize Z exponent (-1, +1].\n    if isinstance(z, numbers.Real):\n        z %= 2\n        if z > 1:\n            z -= 2\n\n    # Canonicalize axis phase exponent into (-0.5, +0.5].\n    if isinstance(a, numbers.Real):\n        a %= 2\n        if a > 1:\n            a -= 2\n        if a <= -0.5:\n            a += 1\n            x = -x\n        elif a > 0.5:\n            a -= 1\n            x = -x\n    '
    t = f(3, 0, 0)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0
    t = f(1.5, 0, 0)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0
    t = f(0, 3, 0)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == 1
    assert t.axis_phase_exponent == 0
    t = f(0, 1.5, 0)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == -0.5
    assert t.axis_phase_exponent == 0
    t = f(0.5, 0, 2.25)._canonical()
    assert t.x_exponent == 0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0.25
    t = f(0.5, 0, 1.25)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0.25
    t = f(0.5, 0, 0.75)._canonical()
    assert t.x_exponent == -0.5
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == -0.25
    t = f(1, 1, 0.5)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == 0
    t = f(1, 0.25, 0.5)._canonical()
    assert t.x_exponent == 1
    assert t.z_exponent == 0
    assert t.axis_phase_exponent == -0.375
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(t), cirq.unitary(f(1, 0.25, 0.5)), atol=1e-08)
    t = f(0, 0.25, 0.5)._canonical()
    assert t.x_exponent == 0
    assert t.z_exponent == 0.25
    assert t.axis_phase_exponent == 0