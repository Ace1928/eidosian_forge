import itertools
import math
from typing import List
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_mutable_pauli_string_equality():
    eq = cirq.testing.EqualsTester()
    a, b, c = cirq.LineQubit.range(3)
    eq.add_equality_group(cirq.MutablePauliString(), cirq.MutablePauliString(), cirq.MutablePauliString(1), cirq.MutablePauliString(-1, -1), cirq.MutablePauliString({a: 0}), cirq.MutablePauliString({a: 'I'}), cirq.MutablePauliString({a: cirq.I}), cirq.MutablePauliString(cirq.I(a)), cirq.MutablePauliString(cirq.I(b)))
    eq.add_equality_group(cirq.MutablePauliString({a: 'X'}), cirq.MutablePauliString({a: 1}), cirq.MutablePauliString({a: cirq.X}), cirq.MutablePauliString(cirq.X(a)))
    eq.add_equality_group(cirq.MutablePauliString({b: 'X'}), cirq.MutablePauliString({b: 1}), cirq.MutablePauliString({b: cirq.X}), cirq.MutablePauliString(cirq.X(b)), cirq.MutablePauliString(-1j, cirq.Y(b), cirq.Z(b)))
    eq.add_equality_group(cirq.MutablePauliString({a: 'X', b: 'Y', c: 'Z'}), cirq.MutablePauliString({a: 1, b: 2, c: 3}), cirq.MutablePauliString({a: cirq.X, b: cirq.Y, c: cirq.Z}), cirq.MutablePauliString(cirq.X(a) * cirq.Y(b) * cirq.Z(c)), cirq.MutablePauliString(cirq.MutablePauliString(cirq.X(a) * cirq.Y(b) * cirq.Z(c))), cirq.MutablePauliString(cirq.MutablePauliString(cirq.X(a), cirq.Y(b), cirq.Z(c))))
    p = cirq.X(a) * cirq.Y(b)
    assert p == cirq.MutablePauliString(p)
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.MutablePauliString('test')
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.MutablePauliString(object())