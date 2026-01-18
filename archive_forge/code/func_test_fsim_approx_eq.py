import numpy as np
import pytest
import sympy
import cirq
def test_fsim_approx_eq():
    assert cirq.approx_eq(cirq.FSimGate(1, 2), cirq.FSimGate(1.00001, 2.00001), atol=0.01)