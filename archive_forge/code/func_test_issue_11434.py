from sympy.core.numbers import I
from sympy.core.symbol import symbols
from sympy.matrices.common import _MinimalMatrix, _CastableMatrix
from sympy.matrices.matrices import MatrixReductions
from sympy.testing.pytest import raises
from sympy.matrices import Matrix, zeros
from sympy.core.symbol import Symbol
from sympy.core.numbers import Rational
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.simplify.simplify import simplify
from sympy.abc import x
def test_issue_11434():
    ax, ay, bx, by, cx, cy, dx, dy, ex, ey, t0, t1 = symbols('a_x a_y b_x b_y c_x c_y d_x d_y e_x e_y t_0 t_1')
    M = Matrix([[ax, ay, ax * t0, ay * t0, 0], [bx, by, bx * t0, by * t0, 0], [cx, cy, cx * t0, cy * t0, 1], [dx, dy, dx * t0, dy * t0, 1], [ex, ey, 2 * ex * t1 - ex * t0, 2 * ey * t1 - ey * t0, 0]])
    assert M.rank() == 4