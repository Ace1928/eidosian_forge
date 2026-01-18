from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import H, XGate, IdentityGate
from sympy.physics.quantum.operator import Operator, IdentityOperator
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.spin import Jx, Jy, Jz, Jplus, Jminus, J2, JzKet
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.state import Ket
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.qubit import Qubit, QubitBra
from sympy.physics.quantum.boson import BosonOp, BosonFockKet, BosonFockBra
def test_issue24158_ket_times_op():
    P = BosonFockKet(0) * BosonOp('a')
    assert qapply(P) == P
    P = Qubit(1) * XGate(0)
    assert qapply(P) == P
    P1 = Mul(QubitBra(0), Mul(QubitBra(0), Qubit(0)), XGate(0))
    assert qapply(P1) == QubitBra(0) * XGate(0)
    P1 = qapply(P1, dagger=True)
    assert qapply(P1, dagger=True) == QubitBra(1)
    P2 = QubitBra(0) * QubitBra(0) * Qubit(0) * XGate(0)
    P2 = qapply(P2, dagger=True)
    assert qapply(P2, dagger=True) == QubitBra(1)
    assert qapply(QubitBra(1) * IdentityOperator()) == QubitBra(1)
    assert qapply(IdentityGate(0) * (Qubit(0) + Qubit(1))) == Qubit(0) + Qubit(1)