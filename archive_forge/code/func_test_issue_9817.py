from sympy.functions import adjoint, conjugate, transpose
from sympy.matrices.expressions import MatrixSymbol, Adjoint, trace, Transpose
from sympy.matrices import eye, Matrix
from sympy.assumptions.ask import Q
from sympy.assumptions.refine import refine
from sympy.core.singleton import S
from sympy.core.symbol import symbols
def test_issue_9817():
    from sympy.matrices.expressions import Identity
    v = MatrixSymbol('v', 3, 1)
    A = MatrixSymbol('A', 3, 3)
    x = Matrix([i + 1 for i in range(3)])
    X = Identity(3)
    quadratic = v.T * A * v
    subbed = quadratic.xreplace({v: x, A: X})
    assert subbed.as_explicit() == Matrix([[14]])