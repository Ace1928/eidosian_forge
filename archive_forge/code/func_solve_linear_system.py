from __future__ import annotations
from sympy.core import (S, Add, Symbol, Dummy, Expr, Mul)
from sympy.core.assumptions import check_assumptions
from sympy.core.exprtools import factor_terms
from sympy.core.function import (expand_mul, expand_log, Derivative,
from sympy.core.logic import fuzzy_not
from sympy.core.numbers import ilcm, Float, Rational, _illegal
from sympy.core.power import integer_log, Pow
from sympy.core.relational import Eq, Ne
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.sympify import sympify, _sympify
from sympy.core.traversal import preorder_traversal
from sympy.logic.boolalg import And, BooleanAtom
from sympy.functions import (log, exp, LambertW, cos, sin, tan, acos, asin, atan,
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.piecewise import piecewise_fold, Piecewise
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.integrals.integrals import Integral
from sympy.ntheory.factor_ import divisors
from sympy.simplify import (simplify, collect, powsimp, posify,  # type: ignore
from sympy.simplify.sqrtdenest import sqrt_depth
from sympy.simplify.fu import TR1, TR2i
from sympy.matrices.common import NonInvertibleMatrixError
from sympy.matrices import Matrix, zeros
from sympy.polys import roots, cancel, factor, Poly
from sympy.polys.polyerrors import GeneratorsNeeded, PolynomialError
from sympy.polys.solvers import sympy_eqs_to_ring, solve_lin_sys
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import filldedent, debugf
from sympy.utilities.iterables import (connected_components,
from sympy.utilities.decorator import conserve_mpmath_dps
from mpmath import findroot
from sympy.solvers.polysys import solve_poly_system
from types import GeneratorType
from collections import defaultdict
from itertools import combinations, product
import warnings
from sympy.solvers.bivariate import (
def solve_linear_system(system, *symbols, **flags):
    """
    Solve system of $N$ linear equations with $M$ variables, which means
    both under- and overdetermined systems are supported.

    Explanation
    ===========

    The possible number of solutions is zero, one, or infinite. Respectively,
    this procedure will return None or a dictionary with solutions. In the
    case of underdetermined systems, all arbitrary parameters are skipped.
    This may cause a situation in which an empty dictionary is returned.
    In that case, all symbols can be assigned arbitrary values.

    Input to this function is a $N\\times M + 1$ matrix, which means it has
    to be in augmented form. If you prefer to enter $N$ equations and $M$
    unknowns then use ``solve(Neqs, *Msymbols)`` instead. Note: a local
    copy of the matrix is made by this routine so the matrix that is
    passed will not be modified.

    The algorithm used here is fraction-free Gaussian elimination,
    which results, after elimination, in an upper-triangular matrix.
    Then solutions are found using back-substitution. This approach
    is more efficient and compact than the Gauss-Jordan method.

    Examples
    ========

    >>> from sympy import Matrix, solve_linear_system
    >>> from sympy.abc import x, y

    Solve the following system::

           x + 4 y ==  2
        -2 x +   y == 14

    >>> system = Matrix(( (1, 4, 2), (-2, 1, 14)))
    >>> solve_linear_system(system, x, y)
    {x: -6, y: 2}

    A degenerate system returns an empty dictionary:

    >>> system = Matrix(( (0,0,0), (0,0,0) ))
    >>> solve_linear_system(system, x, y)
    {}

    """
    assert system.shape[1] == len(symbols) + 1
    eqs = list(system * Matrix(symbols + (-1,)))
    eqs, ring = sympy_eqs_to_ring(eqs, symbols)
    sol = solve_lin_sys(eqs, ring, _raw=False)
    if sol is not None:
        sol = {sym: val for sym, val in sol.items() if sym != val}
    return sol