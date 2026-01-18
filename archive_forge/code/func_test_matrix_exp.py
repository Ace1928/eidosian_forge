from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.hyperbolic import sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.core.containers import Tuple
from sympy.functions import exp, cos, sin, log, Ci, Si, erf, erfi
from sympy.matrices import dotprodsimp, NonSquareMatrixError
from sympy.solvers.ode import dsolve
from sympy.solvers.ode.ode import constant_renumber
from sympy.solvers.ode.subscheck import checksysodesol
from sympy.solvers.ode.systems import (_classify_linear_system, linear_ode_to_matrix,
from sympy.functions import airyai, airybi
from sympy.integrals.integrals import Integral
from sympy.simplify.ratsimp import ratsimp
from sympy.testing.pytest import ON_CI, raises, slow, skip, XFAIL
def test_matrix_exp():
    from sympy.matrices.dense import Matrix, eye, zeros
    from sympy.solvers.ode.systems import matrix_exp
    t = Symbol('t')
    for n in range(1, 6 + 1):
        assert matrix_exp(zeros(n), t) == eye(n)
    for n in range(1, 6 + 1):
        A = eye(n)
        expAt = exp(t) * eye(n)
        assert matrix_exp(A, t) == expAt
    for n in range(1, 6 + 1):
        A = Matrix(n, n, lambda i, j: i + 1 if i == j else 0)
        expAt = Matrix(n, n, lambda i, j: exp((i + 1) * t) if i == j else 0)
        assert matrix_exp(A, t) == expAt
    A = Matrix([[0, 1], [-1, 0]])
    expAt = Matrix([[cos(t), sin(t)], [-sin(t), cos(t)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[2, -5], [2, -4]])
    expAt = Matrix([[3 * exp(-t) * sin(t) + exp(-t) * cos(t), -5 * exp(-t) * sin(t)], [2 * exp(-t) * sin(t), -3 * exp(-t) * sin(t) + exp(-t) * cos(t)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[21, 17, 6], [-5, -1, -6], [4, 4, 16]])
    expAt = Matrix([[2 * t * exp(16 * t) + 5 * exp(16 * t) / 4 - exp(4 * t) / 4, 2 * t * exp(16 * t) + 5 * exp(16 * t) / 4 - 5 * exp(4 * t) / 4, exp(16 * t) / 2 - exp(4 * t) / 2], [-2 * t * exp(16 * t) - exp(16 * t) / 4 + exp(4 * t) / 4, -2 * t * exp(16 * t) - exp(16 * t) / 4 + 5 * exp(4 * t) / 4, -exp(16 * t) / 2 + exp(4 * t) / 2], [4 * t * exp(16 * t), 4 * t * exp(16 * t), exp(16 * t)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, -S(1) / 8], [0, 0, S(1) / 2, S(1) / 2]])
    expAt = Matrix([[exp(t), t * exp(t), 4 * t * exp(3 * t / 4) + 8 * t * exp(t) + 48 * exp(3 * t / 4) - 48 * exp(t), -2 * t * exp(3 * t / 4) - 2 * t * exp(t) - 16 * exp(3 * t / 4) + 16 * exp(t)], [0, exp(t), -t * exp(3 * t / 4) - 8 * exp(3 * t / 4) + 8 * exp(t), t * exp(3 * t / 4) / 2 + 2 * exp(3 * t / 4) - 2 * exp(t)], [0, 0, t * exp(3 * t / 4) / 4 + exp(3 * t / 4), -t * exp(3 * t / 4) / 8], [0, 0, t * exp(3 * t / 4) / 2, -t * exp(3 * t / 4) / 4 + exp(3 * t / 4)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
    expAt = Matrix([[cos(t), sin(t), 0, 0], [-sin(t), cos(t), 0, 0], [0, 0, cos(t), sin(t)], [0, 0, -sin(t), cos(t)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[0, 1, 1, 0], [-1, 0, 0, 1], [0, 0, 0, 1], [0, 0, -1, 0]])
    expAt = Matrix([[cos(t), sin(t), t * cos(t), t * sin(t)], [-sin(t), cos(t), -t * sin(t), t * cos(t)], [0, 0, cos(t), sin(t)], [0, 0, -sin(t), cos(t)]])
    assert matrix_exp(A, t) == expAt
    A = Matrix([[0, I], [I, 0]])
    expAt = Matrix([[exp(I * t) / 2 + exp(-I * t) / 2, exp(I * t) / 2 - exp(-I * t) / 2], [exp(I * t) / 2 - exp(-I * t) / 2, exp(I * t) / 2 + exp(-I * t) / 2]])
    assert matrix_exp(A, t) == expAt
    M = Matrix([[1, 2, 3], [4, 5, 6], [7, 7, 7]])
    M1 = Matrix([[t, 1], [1, 1]])
    raises(ValueError, lambda: matrix_exp(M[:, :2], t))
    raises(ValueError, lambda: matrix_exp(M[:2, :], t))
    raises(ValueError, lambda: matrix_exp(M1, t))
    raises(ValueError, lambda: matrix_exp(M1[:1, :1], t))