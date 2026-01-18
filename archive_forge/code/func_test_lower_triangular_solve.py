from sympy.core.numbers import (Float, I, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.polys.polytools import PurePoly
from sympy.matrices import \
from sympy.testing.pytest import raises
def test_lower_triangular_solve():
    raises(NonSquareMatrixError, lambda: SparseMatrix([[1, 2]]).lower_triangular_solve(Matrix([[1, 2]])))
    raises(ShapeError, lambda: SparseMatrix([[1, 2], [0, 4]]).lower_triangular_solve(Matrix([1])))
    raises(ValueError, lambda: SparseMatrix([[1, 2], [3, 4]]).lower_triangular_solve(Matrix([[1, 2], [3, 4]])))
    a, b, c, d = symbols('a:d')
    u, v, w, x = symbols('u:x')
    A = SparseMatrix([[a, 0], [c, d]])
    B = MutableSparseMatrix([[u, v], [w, x]])
    C = ImmutableSparseMatrix([[u, v], [w, x]])
    sol = Matrix([[u / a, v / a], [(w - c * u / a) / d, (x - c * v / a) / d]])
    assert A.lower_triangular_solve(B) == sol
    assert A.lower_triangular_solve(C) == sol