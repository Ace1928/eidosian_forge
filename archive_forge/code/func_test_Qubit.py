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
def test_Qubit():
    array = [0, 0, 1, 1, 0]
    qb = Qubit('00110')
    assert qb.flip(0) == Qubit('00111')
    assert qb.flip(1) == Qubit('00100')
    assert qb.flip(4) == Qubit('10110')
    assert qb.qubit_values == (0, 0, 1, 1, 0)
    assert qb.dimension == 5
    for i in range(5):
        assert qb[i] == array[4 - i]
    assert len(qb) == 5
    qb = Qubit('110')