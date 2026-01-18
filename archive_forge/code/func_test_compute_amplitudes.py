import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_compute_amplitudes():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.X(a), cirq.H(a), cirq.H(b))
    sim = cirq.Simulator()
    result = sim.compute_amplitudes(c, [0])
    np.testing.assert_allclose(np.array(result), np.array([0.5]))
    result = sim.compute_amplitudes(c, [1, 2, 3])
    np.testing.assert_allclose(np.array(result), np.array([0.5, -0.5, -0.5]))
    result = sim.compute_amplitudes(c, (1, 2, 3), qubit_order=(b, a))
    np.testing.assert_allclose(np.array(result), np.array([-0.5, 0.5, -0.5]))