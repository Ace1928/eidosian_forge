import random
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_measure_partial():
    state = Qubit('01') + Qubit('10')
    assert measure_partial(state, (0,)) == [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, int(0)) == [(Qubit('10'), S.Half), (Qubit('01'), S.Half)]
    assert measure_partial(state, (0,)) == measure_partial(state, (1,))[::-1]
    state1 = sqrt(2) / sqrt(3) * Qubit('00001') + 1 / sqrt(3) * Qubit('11111')
    assert measure_partial(state1, (0,)) == [(sqrt(2) / sqrt(3) * Qubit('00001') + 1 / sqrt(3) * Qubit('11111'), 1)]
    assert measure_partial(state1, (1, 2)) == measure_partial(state1, (3, 4))
    assert measure_partial(state1, (1, 2, 3)) == [(Qubit('00001'), Rational(2, 3)), (Qubit('11111'), Rational(1, 3))]
    state2 = Qubit('1111') + Qubit('1101') + Qubit('1011') + Qubit('1000')
    assert measure_partial(state2, (0, 1, 3)) == [(Qubit('1000'), Rational(1, 4)), (Qubit('1101'), Rational(1, 4)), (Qubit('1011') / sqrt(2) + Qubit('1111') / sqrt(2), S.Half)]
    assert measure_partial(state2, (0,)) == [(Qubit('1000'), Rational(1, 4)), (Qubit('1111') / sqrt(3) + Qubit('1101') / sqrt(3) + Qubit('1011') / sqrt(3), Rational(3, 4))]