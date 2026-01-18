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
def test_eval_trace():
    q1 = Qubit('10110')
    q2 = Qubit('01010')
    d = Density([q1, 0.6], [q2, 0.4])
    t = Tr(d)
    assert t.doit() == 1.0
    t = Tr(d, 0)
    assert t.doit() == 0.4 * Density([Qubit('0101'), 1]) + 0.6 * Density([Qubit('1011'), 1])
    t = Tr(d, 4)
    assert t.doit() == 0.4 * Density([Qubit('1010'), 1]) + 0.6 * Density([Qubit('0110'), 1])
    t = Tr(d, 2)
    assert t.doit() == 0.4 * Density([Qubit('0110'), 1]) + 0.6 * Density([Qubit('1010'), 1])
    t = Tr(d, [0, 1, 2, 3, 4])
    assert t.doit() == 1.0
    t = Tr(d, [2, 1, 3])
    assert t.doit() == 0.4 * Density([Qubit('00'), 1]) + 0.6 * Density([Qubit('10'), 1])
    q = 1 / sqrt(2) * (Qubit('00') + Qubit('11'))
    d = Density([q, 1.0])
    t = Tr(d, 0)
    assert t.doit() == 0.5 * Density([Qubit('0'), 1]) + 0.5 * Density([Qubit('1'), 1])