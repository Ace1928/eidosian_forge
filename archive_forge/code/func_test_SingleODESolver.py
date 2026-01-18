from sympy.core.function import (Derivative, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, symbols)
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (asinh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (Ei, erfi)
from sympy.functions.special.hyper import hyper
from sympy.integrals.integrals import (Integral, integrate)
from sympy.polys.rootoftools import rootof
from sympy.core import Function, Symbol
from sympy.functions import airyai, airybi, besselj, bessely, lowergamma
from sympy.integrals.risch import NonElementaryIntegral
from sympy.solvers.ode import classify_ode, dsolve
from sympy.solvers.ode.ode import allhints, _remove_redundant_solutions
from sympy.solvers.ode.single import (FirstLinear, ODEMatchError,
from sympy.solvers.ode.subscheck import checkodesol
from sympy.testing.pytest import raises, slow, ON_CI
import traceback
from sympy.solvers.ode.tests.test_single import _test_an_example
def test_SingleODESolver():
    problem = SingleODEProblem(f(x).diff(x), f(x), x)
    solver = SingleODESolver(problem)
    raises(NotImplementedError, lambda: solver.matches())
    raises(NotImplementedError, lambda: solver.get_general_solution())
    raises(NotImplementedError, lambda: solver._matches())
    raises(NotImplementedError, lambda: solver._get_general_solution())
    problem = SingleODEProblem(f(x).diff(x) + f(x) * f(x), f(x), x)
    solver = FirstLinear(problem)
    raises(ODEMatchError, lambda: solver.get_general_solution())
    solver = FirstLinear(problem)
    assert solver.matches() is False
    problem = SingleODEProblem(f(x).diff(x) + f(x), f(x), x)
    assert problem.order == 1
    problem = SingleODEProblem(f(x).diff(x, 4) + f(x).diff(x, 2) - f(x).diff(x, 3), f(x), x)
    assert problem.order == 4
    problem = SingleODEProblem(f(x).diff(x, 3) + f(x).diff(x, 2) - f(x) ** 2, f(x), x)
    assert problem.is_autonomous == True
    problem = SingleODEProblem(f(x).diff(x, 3) + x * f(x).diff(x, 2) - f(x) ** 2, f(x), x)
    assert problem.is_autonomous == False