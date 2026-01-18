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
def test_linodesolve():
    t, x, a = symbols('t x a')
    f, g, h = symbols('f g h', cls=Function)
    raises(ValueError, lambda: linodesolve(1, t))
    raises(ValueError, lambda: linodesolve(a, t))
    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t))
    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t))
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t)), Eq(g(t).diff(t), f(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [C1 * (-Rational(1, 2) + sqrt(5) / 2) * exp(t * (-Rational(1, 2) + sqrt(5) / 2)) + C2 * (-sqrt(5) / 2 - Rational(1, 2)) * exp(t * (-sqrt(5) / 2 - Rational(1, 2))), C1 * exp(t * (-Rational(1, 2) + sqrt(5) / 2)) + C2 * exp(t * (-sqrt(5) / 2 - Rational(1, 2)))]
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol
    raises(ValueError, lambda: linodesolve(1, t, b=Matrix([t + 1])))
    raises(ValueError, lambda: linodesolve(a, t, b=Matrix([log(t) + sin(t)])))
    raises(ValueError, lambda: linodesolve(Matrix([7]), t, b=t ** 2))
    raises(ValueError, lambda: linodesolve(Matrix([a + 10]), t, b=log(t) * cos(t)))
    raises(ValueError, lambda: linodesolve(7, t, b=t ** 2))
    raises(ValueError, lambda: linodesolve(a, t, b=log(t) + sin(t)))
    A1 = Matrix([[1, 2], [2, 4], [4, 6]])
    b1 = Matrix([t, 1, t ** 2])
    raises(NonSquareMatrixError, lambda: linodesolve(A1, t, b=b1))
    A2 = Matrix([[1, 2, 1], [3, 1, 2]])
    b2 = Matrix([t, t ** 2])
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, b=b2))
    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1))
    raises(ValueError, lambda: linodesolve(A1[:2, :], t, b=b1[:1]))
    A1 = Matrix([[1, -1], [1, -1]])
    b1 = Matrix([15 * t - 10, -15 * t - 5])
    sol1 = [C1 + C2 * t + C2 - 10 * t ** 3 + 10 * t ** 2 + t * (15 * t ** 2 - 5 * t) - 10 * t, C1 + C2 * t - 10 * t ** 3 - 5 * t ** 2 + t * (15 * t ** 2 - 5 * t) - 5 * t]
    assert constant_renumber(linodesolve(A1, t, b=b1, type='type2', doit=True), variables=[t]) == sol1
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t) + g(t).diff(t), g(t) + t), Eq(g(t).diff(t), f(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [-C1 * exp(-t / 2 + sqrt(5) * t / 2) / 2 + sqrt(5) * C1 * exp(-t / 2 + sqrt(5) * t / 2) / 2 - sqrt(5) * C2 * exp(-sqrt(5) * t / 2 - t / 2) / 2 - C2 * exp(-sqrt(5) * t / 2 - t / 2) / 2 - exp(-t / 2 + sqrt(5) * t / 2) * Integral(t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)) - sqrt(5) * t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)), t) / 2 + sqrt(5) * exp(-t / 2 + sqrt(5) * t / 2) * Integral(t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)) - sqrt(5) * t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)), t) / 2 - sqrt(5) * exp(-sqrt(5) * t / 2 - t / 2) * Integral(-sqrt(5) * t * exp(t / 2 + sqrt(5) * t / 2) / 5, t) / 2 - exp(-sqrt(5) * t / 2 - t / 2) * Integral(-sqrt(5) * t * exp(t / 2 + sqrt(5) * t / 2) / 5, t) / 2, C1 * exp(-t / 2 + sqrt(5) * t / 2) + C2 * exp(-sqrt(5) * t / 2 - t / 2) + exp(-t / 2 + sqrt(5) * t / 2) * Integral(t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)) - sqrt(5) * t * exp(-sqrt(5) * t / 2 + t / 2) / (-5 + sqrt(5)), t) + exp(-sqrt(5) * t / 2 - t / 2) * Integral(-sqrt(5) * t * exp(t / 2 + sqrt(5) * t / 2) / 5, t)]
    assert constant_renumber(linodesolve(A, t, b=b), variables=[t]) == sol
    sol1 = [-C1 * exp(-t / 2 + sqrt(5) * t / 2) / 2 + sqrt(5) * C1 * exp(-t / 2 + sqrt(5) * t / 2) / 2 - sqrt(5) * C2 * exp(-sqrt(5) * t / 2 - t / 2) / 2 - C2 * exp(-sqrt(5) * t / 2 - t / 2) / 2, C1 * exp(-t / 2 + sqrt(5) * t / 2) + C2 * exp(-sqrt(5) * t / 2 - t / 2)]
    assert constant_renumber(linodesolve(A, t, type='type2'), variables=[t]) == sol1
    raises(ValueError, lambda: linodesolve(t + 10, t))
    raises(ValueError, lambda: linodesolve(a * t, t))
    A1 = Matrix([[1, t], [-t, 1]])
    B1, _ = _is_commutative_anti_derivative(A1, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, B=B1))
    raises(ValueError, lambda: linodesolve(A1, t, B=1))
    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    B2, _ = _is_commutative_anti_derivative(A2, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A2, t, B=B2[:2, :]))
    raises(ValueError, lambda: linodesolve(A2, t, B=2))
    raises(ValueError, lambda: linodesolve(A2, t, B=B2, type='type31'))
    raises(ValueError, lambda: linodesolve(A1, t, B=B2))
    raises(ValueError, lambda: linodesolve(A2, t, B=B1))
    func = [f(t), g(t)]
    eq = [Eq(f(t).diff(t), f(t) + t * g(t)), Eq(g(t).diff(t), -t * f(t) + g(t))]
    ceq = canonical_odes(eq, func, t)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, t, 1)
    A = A0
    sol = [(C1 / 2 - I * C2 / 2) * exp(I * t ** 2 / 2 + t) + (C1 / 2 + I * C2 / 2) * exp(-I * t ** 2 / 2 + t), (-I * C1 / 2 + C2 / 2) * exp(-I * t ** 2 / 2 + t) + (I * C1 / 2 + C2 / 2) * exp(I * t ** 2 / 2 + t)]
    assert constant_renumber(linodesolve(A, t), variables=Tuple(*eq).free_symbols) == sol
    assert constant_renumber(linodesolve(A, t, type='type3'), variables=Tuple(*eq).free_symbols) == sol
    A1 = Matrix([[t, 1], [t, -1]])
    raises(NotImplementedError, lambda: linodesolve(A1, t))
    raises(ValueError, lambda: linodesolve(t + 10, t, b=Matrix([t + 1])))
    raises(ValueError, lambda: linodesolve(a * t, t, b=Matrix([log(t) + sin(t)])))
    raises(ValueError, lambda: linodesolve(Matrix([7 * t]), t, b=t ** 2))
    raises(ValueError, lambda: linodesolve(Matrix([a + 10 * log(t)]), t, b=log(t) * cos(t)))
    raises(ValueError, lambda: linodesolve(7 * t, t, b=t ** 2))
    raises(ValueError, lambda: linodesolve(a * t ** 2, t, b=log(t) + sin(t)))
    A1 = Matrix([[1, t], [-t, 1]])
    b1 = Matrix([t, t ** 2])
    B1, _ = _is_commutative_anti_derivative(A1, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A1[:, :1], t, b=b1))
    A2 = Matrix([[t, t, t], [t, t, t], [t, t, t]])
    b2 = Matrix([t, 1, t ** 2])
    B2, _ = _is_commutative_anti_derivative(A2, t)
    raises(NonSquareMatrixError, lambda: linodesolve(A2[:2, :], t, b=b2))
    raises(ValueError, lambda: linodesolve(A1, t, b=b2))
    raises(ValueError, lambda: linodesolve(A2, t, b=b1))
    raises(ValueError, lambda: linodesolve(A1, t, b=b1, B=B2))
    raises(ValueError, lambda: linodesolve(A2, t, b=b2, B=B1))
    func = [f(x), g(x), h(x)]
    eq = [Eq(f(x).diff(x), x * (f(x) + g(x) + h(x)) + x), Eq(g(x).diff(x), x * (f(x) + g(x) + h(x)) + x), Eq(h(x).diff(x), x * (f(x) + g(x) + h(x)) + 1)]
    ceq = canonical_odes(eq, func, x)
    (A1, A0), b = linear_ode_to_matrix(ceq[0], func, x, 1)
    A = A0
    _x1 = exp(-3 * x ** 2 / 2)
    _x2 = exp(3 * x ** 2 / 2)
    _x3 = Integral(2 * _x1 * x / 3 + _x1 / 3 + x / 3 - Rational(1, 3), x)
    _x4 = 2 * _x2 * _x3 / 3
    _x5 = Integral(2 * _x1 * x / 3 + _x1 / 3 - 2 * x / 3 + Rational(2, 3), x)
    sol = [C1 * _x2 / 3 - C1 / 3 + C2 * _x2 / 3 - C2 / 3 + C3 * _x2 / 3 + 2 * C3 / 3 + _x2 * _x5 / 3 + _x3 / 3 + _x4 - _x5 / 3, C1 * _x2 / 3 + 2 * C1 / 3 + C2 * _x2 / 3 - C2 / 3 + C3 * _x2 / 3 - C3 / 3 + _x2 * _x5 / 3 + _x3 / 3 + _x4 - _x5 / 3, C1 * _x2 / 3 - C1 / 3 + C2 * _x2 / 3 + 2 * C2 / 3 + C3 * _x2 / 3 - C3 / 3 + _x2 * _x5 / 3 - 2 * _x3 / 3 + _x4 + 2 * _x5 / 3]
    assert constant_renumber(linodesolve(A, x, b=b), variables=Tuple(*eq).free_symbols) == sol
    assert constant_renumber(linodesolve(A, x, b=b, type='type4'), variables=Tuple(*eq).free_symbols) == sol
    A1 = Matrix([[t, 1], [t, -1]])
    raises(NotImplementedError, lambda: linodesolve(A1, t, b=b1))
    sol1 = [-C1 / 3 - C2 / 3 + 2 * C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2), 2 * C1 / 3 - C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2), -C1 / 3 + 2 * C2 / 3 - C3 / 3 + (C1 / 3 + C2 / 3 + C3 / 3) * exp(3 * x ** 2 / 2)]
    assert constant_renumber(linodesolve(A, x, type='type4', doit=True), variables=Tuple(*eq).free_symbols) == sol1