from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.symbol import (Wild, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices import Matrix, ImmutableMatrix
from sympy.physics.quantum.gate import (XGate, YGate, ZGate, random_circuit,
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit, IntQubit, qubit_to_matrix, \
from sympy.physics.quantum.matrixutils import matrix_to_zero
from sympy.physics.quantum.matrixcache import sqrt2_inv
from sympy.physics.quantum import Dagger
def test_cnot_commutators():
    """Test commutators of involving CNOT gates."""
    assert Commutator(CNOT(0, 1), Z(0)).doit() == 0
    assert Commutator(CNOT(0, 1), T(0)).doit() == 0
    assert Commutator(CNOT(0, 1), S(0)).doit() == 0
    assert Commutator(CNOT(0, 1), X(1)).doit() == 0
    assert Commutator(CNOT(0, 1), CNOT(0, 1)).doit() == 0
    assert Commutator(CNOT(0, 1), CNOT(0, 2)).doit() == 0
    assert Commutator(CNOT(0, 2), CNOT(0, 1)).doit() == 0
    assert Commutator(CNOT(1, 2), CNOT(1, 0)).doit() == 0