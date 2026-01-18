from sympy.core.function import expand_mul
from sympy.core.numbers import (I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.simplify.simplify import simplify
from sympy.matrices.matrices import (ShapeError, NonSquareMatrixError)
from sympy.matrices import (
from sympy.testing.pytest import raises
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.solvers.solveset import linsolve
from sympy.abc import x, y
def test_issue_17247_expression_blowup_30():
    M = Matrix(S('[\n        [             -3/4,       45/32 - 37*I/16,                   0,                     0],\n        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],\n        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],\n        [                0,                     0,                   0, -177/128 - 1369*I/128]]'))
    with dotprodsimp(True):
        assert M.cholesky_solve(ones(4, 1)) == Matrix(S('[\n            [                          -32549314808672/3306971225785 - 17397006745216*I/3306971225785],\n            [                               67439348256/3306971225785 - 9167503335872*I/3306971225785],\n            [-15091965363354518272/21217636514687010905 + 16890163109293858304*I/21217636514687010905],\n            [                                                          -11328/952745 + 87616*I/952745]]'))