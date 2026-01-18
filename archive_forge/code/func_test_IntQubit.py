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
def test_IntQubit():
    iqb = IntQubit(0, nqubits=1)
    assert qubit_to_matrix(Qubit('0')) == qubit_to_matrix(iqb)
    qb = Qubit('1010')
    assert qubit_to_matrix(IntQubit(qb)) == qubit_to_matrix(qb)
    iqb = IntQubit(1, nqubits=1)
    assert qubit_to_matrix(Qubit('1')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(1)) == qubit_to_matrix(iqb)
    iqb = IntQubit(7, nqubits=4)
    assert qubit_to_matrix(Qubit('0111')) == qubit_to_matrix(iqb)
    assert qubit_to_matrix(IntQubit(7, 4)) == qubit_to_matrix(iqb)
    iqb = IntQubit(8)
    assert iqb.as_int() == 8
    assert iqb.qubit_values == (1, 0, 0, 0)
    iqb = IntQubit(7, 4)
    assert iqb.qubit_values == (0, 1, 1, 1)
    assert IntQubit(3) == IntQubit(3, 2)
    iqb = IntQubit(3)
    iqb_bra = IntQubitBra(3)
    assert iqb.dual_class() == IntQubitBra
    assert iqb_bra.dual_class() == IntQubit
    iqb = IntQubit(5)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(1)
    iqb = IntQubit(4)
    iqb_bra = IntQubitBra(5)
    assert iqb._eval_innerproduct_IntQubitBra(iqb_bra) == Integer(0)
    raises(ValueError, lambda: IntQubit(4, 1))
    raises(ValueError, lambda: IntQubit('5'))
    raises(ValueError, lambda: IntQubit(5, '5'))
    raises(ValueError, lambda: IntQubit(5, nqubits='5'))
    raises(TypeError, lambda: IntQubit(5, bad_arg=True))