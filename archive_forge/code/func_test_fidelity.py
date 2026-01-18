from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import log
from sympy.external import import_module
from sympy.physics.quantum.density import Density, entropy, fidelity
from sympy.physics.quantum.state import Ket, TimeDepKet
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.cartesian import XKet, PxKet, PxOp, XOp
from sympy.physics.quantum.spin import JzKet
from sympy.physics.quantum.operator import OuterProduct
from sympy.physics.quantum.trace import Tr
from sympy.functions import sqrt
from sympy.testing.pytest import raises
from sympy.physics.quantum.matrixutils import scipy_sparse_matrix
from sympy.physics.quantum.tensorproduct import TensorProduct
def test_fidelity():
    up = JzKet(S.Half, S.Half)
    down = JzKet(S.Half, Rational(-1, 2))
    updown = S.One / sqrt(2) * up + S.One / sqrt(2) * down
    up_dm = represent(up * Dagger(up))
    down_dm = represent(down * Dagger(down))
    updown_dm = represent(updown * Dagger(updown))
    assert abs(fidelity(up_dm, up_dm) - 1) < 0.001
    assert fidelity(up_dm, down_dm) < 0.001
    assert abs(fidelity(up_dm, updown_dm) - S.One / sqrt(2)) < 0.001
    assert abs(fidelity(updown_dm, down_dm) - S.One / sqrt(2)) < 0.001
    up_dm = Density([up, 1.0])
    down_dm = Density([down, 1.0])
    updown_dm = Density([updown, 1.0])
    assert abs(fidelity(up_dm, up_dm) - 1) < 0.001
    assert abs(fidelity(up_dm, down_dm)) < 0.001
    assert abs(fidelity(up_dm, updown_dm) - S.One / sqrt(2)) < 0.001
    assert abs(fidelity(updown_dm, down_dm) - S.One / sqrt(2)) < 0.001
    updown2 = sqrt(3) / 2 * up + S.Half * down
    d1 = Density([updown, 0.25], [updown2, 0.75])
    d2 = Density([updown, 0.75], [updown2, 0.25])
    assert abs(fidelity(d1, d2) - 0.991) < 0.001
    assert abs(fidelity(d2, d1) - fidelity(d1, d2)) < 0.001
    state1 = Qubit('0')
    state2 = Qubit('1')
    state3 = S.One / sqrt(2) * state1 + S.One / sqrt(2) * state2
    state4 = sqrt(Rational(2, 3)) * state1 + S.One / sqrt(3) * state2
    state1_dm = Density([state1, 1])
    state2_dm = Density([state2, 1])
    state3_dm = Density([state3, 1])
    assert fidelity(state1_dm, state1_dm) == 1
    assert fidelity(state1_dm, state2_dm) == 0
    assert abs(fidelity(state1_dm, state3_dm) - 1 / sqrt(2)) < 0.001
    assert abs(fidelity(state3_dm, state2_dm) - 1 / sqrt(2)) < 0.001
    d1 = Density([state3, 0.7], [state4, 0.3])
    d2 = Density([state3, 0.2], [state4, 0.8])
    assert abs(fidelity(d1, d1) - 1) < 0.001
    assert abs(fidelity(d1, d2) - 0.996) < 0.001
    assert abs(fidelity(d1, d2) - fidelity(d2, d1)) < 0.001
    mat1 = [[0, 0], [0, 0], [0, 0]]
    mat2 = [[0, 0], [0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))
    mat1 = [[0, 0], [0, 0]]
    mat2 = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    raises(ValueError, lambda: fidelity(mat1, mat2))
    x, y = (1, 2)
    raises(ValueError, lambda: fidelity(x, y))