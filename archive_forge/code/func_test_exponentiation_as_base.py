import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_exponentiation_as_base():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    with pytest.raises(TypeError, match='unsupported'):
        _ = (2 * p) ** 5
    with pytest.raises(TypeError, match='unsupported'):
        _ = p ** 'test'
    with pytest.raises(TypeError, match='unsupported'):
        _ = p ** 1j
    assert p ** (-1) == p
    assert cirq.approx_eq(p ** 0.5, cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=0))
    assert cirq.approx_eq(p ** (-0.5), cirq.PauliStringPhasor(p, exponent_neg=-0.5, exponent_pos=0))
    assert cirq.approx_eq(math.e ** (0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))
    assert cirq.approx_eq(2 ** (0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25 * math.log(2), exponent_pos=0.25 * math.log(2)))
    assert cirq.approx_eq(np.exp(0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))
    np.testing.assert_allclose(cirq.unitary(np.exp(0.5j * math.pi * cirq.Z(a))), np.diag([np.exp(0.5j * math.pi), np.exp(-0.5j * math.pi)]), atol=1e-08)