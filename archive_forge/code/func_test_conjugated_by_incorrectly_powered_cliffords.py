import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_conjugated_by_incorrectly_powered_cliffords():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString([cirq.X(a), cirq.Z(b)])
    cliffords = [cirq.H(a), cirq.X(a), cirq.Y(a), cirq.Z(a), cirq.H(a), cirq.CNOT(a, b), cirq.CZ(a, b), cirq.SWAP(a, b), cirq.ISWAP(a, b), cirq.XX(a, b), cirq.YY(a, b), cirq.ZZ(a, b)]
    for c in cliffords:
        with pytest.raises(TypeError, match='not a known Clifford'):
            _ = p.conjugated_by(c ** 0.1)
        with pytest.raises(TypeError, match='not a known Clifford'):
            _ = p.conjugated_by(c ** sympy.Symbol('t'))