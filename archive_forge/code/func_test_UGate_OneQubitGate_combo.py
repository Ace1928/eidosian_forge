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
def test_UGate_OneQubitGate_combo():
    v, w, f, g = symbols('v w f g')
    uMat1 = ImmutableMatrix([[v, w], [f, g]])
    cMat1 = Matrix([[v, w + 1, 0, 0], [f + 1, g, 0, 0], [0, 0, v, w + 1], [0, 0, f + 1, g]])
    u1 = X(0) + UGate(0, uMat1)
    assert represent(u1, nqubits=2) == cMat1
    uMat2 = ImmutableMatrix([[1 / sqrt(2), 1 / sqrt(2)], [I / sqrt(2), -I / sqrt(2)]])
    cMat2_1 = Matrix([[Rational(1, 2) + I / 2, Rational(1, 2) - I / 2], [Rational(1, 2) - I / 2, Rational(1, 2) + I / 2]])
    cMat2_2 = Matrix([[1, 0], [0, I]])
    u2 = UGate(0, uMat2)
    assert represent(H(0) * u2, nqubits=1) == cMat2_1
    assert represent(u2 * H(0), nqubits=1) == cMat2_2