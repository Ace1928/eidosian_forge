import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mul_strings():
    a, b, c, d = cirq.LineQubit.range(4)
    p1 = cirq.PauliString({a: cirq.X, b: cirq.Y, c: cirq.Z})
    p2 = cirq.PauliString({b: cirq.X, c: cirq.Y, d: cirq.Z})
    assert p1 * p2 == -cirq.PauliString({a: cirq.X, b: cirq.Z, c: cirq.X, d: cirq.Z})
    assert cirq.X(a) * cirq.PauliString({a: cirq.X}) == cirq.PauliString()
    assert cirq.PauliString({a: cirq.X}) * cirq.X(a) == cirq.PauliString()
    assert cirq.X(a) * cirq.X(a) == cirq.PauliString()
    assert -cirq.X(a) * -cirq.X(a) == cirq.PauliString()
    with pytest.raises(TypeError, match='unsupported'):
        _ = cirq.X(a) * object()
    with pytest.raises(TypeError, match='unsupported'):
        _ = object() * cirq.X(a)
    assert -cirq.X(a) == -cirq.PauliString({a: cirq.X})