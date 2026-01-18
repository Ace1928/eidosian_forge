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
def minsolve_linear_system(system, *symbols, **flags):
    """
    Find a particular solution to a linear system.

    Explanation
    ===========

    In particular, try to find a solution with the minimal possible number
    of non-zero variables using a naive algorithm with exponential complexity.
    If ``quick=True``, a heuristic is used.

    """
    quick = flags.get('quick', False)
    s0 = solve_linear_system(system, *symbols, **flags)
    if not s0 or all((v == 0 for v in s0.values())):
        return s0
    if quick:
        s = solve_linear_system(system, *symbols)

        def update(determined, solution):
            delete = []
            for k, v in solution.items():
                solution[k] = v.subs(determined)
                if not solution[k].free_symbols:
                    delete.append(k)
                    determined[k] = solution[k]
            for k in delete:
                del solution[k]
        determined = {}
        update(determined, s)
        while s:
            k = max((k for k in s.values()), key=lambda x: (len(x.free_symbols), default_sort_key(x)))
            kfree = k.free_symbols
            x = next(reversed(list(ordered(kfree))))
            if len(kfree) != 1:
                determined[x] = S.Zero
            else:
                val = _vsolve(k, x, check=False)[0]
                if not val and (not any((v.subs(x, val) for v in s.values()))):
                    determined[x] = S.One
                else:
                    determined[x] = val
            update(determined, s)
        return determined
    else:
        N = len(symbols)
        bestsol = minsolve_linear_system(system, *symbols, quick=True)
        n0 = len([x for x in bestsol.values() if x != 0])
        for n in range(n0 - 1, 1, -1):
            debugf('minsolve: %s', n)
            thissol = None
            for nonzeros in combinations(range(N), n):
                subm = Matrix([system.col(i).T for i in nonzeros] + [system.col(-1).T]).T
                s = solve_linear_system(subm, *[symbols[i] for i in nonzeros])
                if s and (not all((v == 0 for v in s.values()))):
                    subs = [(symbols[v], S.One) for v in nonzeros]
                    for k, v in s.items():
                        s[k] = v.subs(subs)
                    for sym in symbols:
                        if sym not in s:
                            if symbols.index(sym) in nonzeros:
                                s[sym] = S.One
                            else:
                                s[sym] = S.Zero
                    thissol = s
                    break
            if thissol is None:
                break
            bestsol = thissol
        return bestsol