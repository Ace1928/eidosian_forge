from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (Integer, pi)
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.trigonometric import sin
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.physics.quantum.operator import (Operator, UnitaryOperator,
from sympy.physics.quantum.state import Ket, Bra, Wavefunction
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.spin import JzKet, JzBra
from sympy.physics.quantum.trace import Tr
from sympy.matrices import eye
def test_eval_power():
    from sympy.core import Pow
    from sympy.core.expr import unchanged
    O = Operator('O')
    U = UnitaryOperator('U')
    H = HermitianOperator('H')
    assert O ** (-1) == O.inv()
    assert U ** (-1) == U.inv()
    assert H ** (-1) == H.inv()
    x = symbols('x', commutative=True)
    assert unchanged(Pow, H, x)
    assert H ** x == Pow(H, x)
    assert Pow(H, x) == Pow(H, x, evaluate=False)
    from sympy.physics.quantum.gate import XGate
    X = XGate(0)
    assert unchanged(Pow, X, x)
    assert X ** x == Pow(X, x)
    assert Pow(X, x, evaluate=False) == Pow(X, x)
    n = symbols('n', integer=True, even=True)
    assert X ** n == 1
    n = symbols('n', integer=True, odd=True)
    assert X ** n == X
    n = symbols('n', integer=True)
    assert unchanged(Pow, X, n)
    assert X ** n == Pow(X, n)
    assert Pow(X, n, evaluate=False) == Pow(X, n)
    assert X ** 4 == 1
    assert X ** 7 == X