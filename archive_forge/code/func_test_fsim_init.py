import numpy as np
import pytest
import sympy
import cirq
def test_fsim_init():
    f = cirq.FSimGate(1, 2)
    assert f.theta == 1
    assert f.phi == 2
    f2 = cirq.FSimGate(theta=1, phi=2)
    assert f2.theta == 1
    assert f2.phi == 2
    f3 = cirq.FSimGate(theta=4, phi=-5)
    assert f3.theta == 4 - 2 * np.pi
    assert f3.phi == -5 + 2 * np.pi