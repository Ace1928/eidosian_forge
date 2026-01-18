import numpy as np
import pytest
import sympy
import cirq
def test_fsim_eq():
    eq = cirq.testing.EqualsTester()
    a, b = cirq.LineQubit.range(2)
    eq.add_equality_group(cirq.FSimGate(1, 2), cirq.FSimGate(1, 2))
    eq.add_equality_group(cirq.FSimGate(2, 1))
    eq.add_equality_group(cirq.FSimGate(0, 0))
    eq.add_equality_group(cirq.FSimGate(1, 1))
    eq.add_equality_group(cirq.FSimGate(1, 2).on(a, b), cirq.FSimGate(1, 2).on(b, a))
    eq.add_equality_group(cirq.FSimGate(np.pi, np.pi), cirq.FSimGate(-np.pi, -np.pi))