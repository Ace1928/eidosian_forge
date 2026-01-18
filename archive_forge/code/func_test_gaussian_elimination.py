import numbers
from typing import List
import numpy as np
import pytest
import sympy
import cirq
from cirq.ops.dense_pauli_string import _vectorized_pauli_mul_phase
def test_gaussian_elimination():

    def table(*rows: str) -> List[cirq.MutableDensePauliString]:
        coefs = {'i': 1j, '-': -1, '+': 1}
        return [cirq.MutableDensePauliString(row[1:].replace('.', 'I'), coefficient=coefs[row[0]]) for row in rows]
    f = cirq.MutableDensePauliString.inline_gaussian_elimination
    t = table()
    f(t)
    assert t == table()
    t = table('+X')
    f(t)
    assert t == table('+X')
    t = table('+.X.X', '+Z.Z.', '+X.XX', '+ZZ.Z')
    f(t)
    assert t == table('+X.XX', '+Z.Z.', '+.X.X', '+.ZZZ')
    t = table('+XXX', '+YYY')
    f(t)
    assert t == table('+XXX', 'iZZZ')
    t = table('+XXXX', '+X...', '+..ZZ', '+.ZZ.')
    f(t)
    assert t == table('+X...', '+.XXX', '+.Z.Z', '+..ZZ')
    t = table('+ZZZ.........', '+XX..........', '+X.X.........', '+...ZZZ......', '+...XX.......', '+...X.X......', '+......ZZ....', '+......XX....', '+........ZZ..', '+........XX..', '+..X....X....', '+..Z....Z....', '+.....X..X...', '+.....Z..Z...', '+.X........X.', '+.Z........Z.', '-X...X.......', '+Z...Z.......', '+...X.......X', '+...Z.......Z', '+......X..X..', '+......Z..Z..')
    f(t)
    assert t == table('-X..........X', '+Z........Z.Z', '-.X.........X', '+.Z.........Z', '-..X........X', '+..Z......Z..', '+...X.......X', '+...Z.......Z', '+....X......X', '+....Z....Z.Z', '+.....X.....X', '+.....Z...Z..', '-......X....X', '+......Z..Z..', '-.......X...X', '+.......Z.Z..', '+........X..X', '+........ZZ..', '+.........X.X', '-..........XX', '+..........ZZ', '-............')