import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_exponentiation_as_exponent():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        _ = math.e ** (math.pi * p)
    with pytest.raises(TypeError, match='unsupported'):
        _ = 'test' ** p
    assert cirq.approx_eq(math.e ** (-0.5j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=-0.5))
    assert cirq.approx_eq(math.e ** (0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))
    assert cirq.approx_eq(2 ** (0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25 * math.log(2), exponent_pos=0.25 * math.log(2)))
    assert cirq.approx_eq(np.exp(0.25j * math.pi * p), cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))