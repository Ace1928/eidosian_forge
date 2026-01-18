from sympy.assumptions.ask import (Q, ask)
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core import (GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Rational, oo, pi)
from sympy.core.relational import (Eq, Gt, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.complexes import (Abs, arg, conjugate, im, re)
from sympy.functions.elementary.exponential import (LambertW, exp, log)
from sympy.functions.elementary.hyperbolic import (atanh, cosh, sinh, tanh)
from sympy.functions.elementary.miscellaneous import (cbrt, root, sqrt)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (acos, asin, atan, atan2, cos, sec, sin, tan)
from sympy.functions.special.error_functions import (erf, erfc, erfcinv, erfinv)
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.dense import Matrix
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import Poly
from sympy.printing.str import sstr
from sympy.simplify.radsimp import denom
from sympy.solvers.solvers import (nsolve, solve, solve_linear)
from sympy.core.function import nfloat
from sympy.solvers import solve_linear_system, solve_linear_system_LU, \
from sympy.solvers.bivariate import _filtered_gens, _solve_lambert, _lambert
from sympy.solvers.solvers import _invert, unrad, checksol, posify, _ispow, \
from sympy.physics.units import cm
from sympy.polys.rootoftools import CRootOf
from sympy.testing.pytest import slow, XFAIL, SKIP, raises
from sympy.core.random import verify_numerically as tn
from sympy.abc import a, b, c, d, e, k, h, p, x, y, z, t, q, m, R
def test_minsolve_linear_system():
    pqt = {'quick': True, 'particular': True}
    pqf = {'quick': False, 'particular': True}
    assert solve([x + y - 5, 2 * x - y - 1], **pqt) == {x: 2, y: 3}
    assert solve([x + y - 5, 2 * x - y - 1], **pqf) == {x: 2, y: 3}

    def count(dic):
        return len([x for x in dic.values() if x == 0])
    assert count(solve([x + y + z, y + z + a + t], **pqt)) == 3
    assert count(solve([x + y + z, y + z + a + t], **pqf)) == 3
    assert count(solve([x + y + z, y + z + a], **pqt)) == 1
    assert count(solve([x + y + z, y + z + a], **pqf)) == 2
    A = Matrix([[1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0], [1, 1, 0, 1, 1, 0, 1, 0, 1, 0, -1, -1, 0, 0], [-1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 1, 1, 0, 1], [1, 0, 1, 1, 0, 1, 1, 0, 0, 1, -1, 0, -1, 0], [-1, 0, -1, 0, 0, -1, 0, 0, 0, 0, 1, 0, 1, 1], [-1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1], [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, -1, -1, 0], [0, -1, -1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1, 1], [0, -1, 0, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, -1], [0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 0, 0, -1, -1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0]])
    v = Matrix(symbols('v:14', integer=True))
    B = Matrix([[2], [-2], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
    eqs = A @ v - B
    assert solve(eqs) == []
    assert solve(eqs, particular=True) == []
    assert all((v for v in solve([x + y + z, y + z + a]).values()))
    for _q in (True, False):
        assert not all((v for v in solve([x + y + z, y + z + a], quick=_q, particular=True).values()))
        raises(ValueError, lambda: solve([x + 1], quick=_q))
        raises(ValueError, lambda: solve([x + 1], quick=_q, particular=False))
    raises(ValueError, lambda: solve(x + 1, particular=True))