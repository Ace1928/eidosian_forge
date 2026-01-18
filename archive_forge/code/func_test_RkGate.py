from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qft import QFT, IQFT, RkGate
from sympy.physics.quantum.gate import (ZGate, SwapGate, HadamardGate, CGate,
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
def test_RkGate():
    x = Symbol('x')
    assert RkGate(1, x).k == x
    assert RkGate(1, x).targets == (1,)
    assert RkGate(1, 1) == ZGate(1)
    assert RkGate(2, 2) == PhaseGate(2)
    assert RkGate(3, 3) == TGate(3)
    assert represent(RkGate(0, x), nqubits=1) == Matrix([[1, 0], [0, exp(2 * I * pi / 2 ** x)]])