from sympy.core import Add, Mul, S
from sympy.core.containers import Tuple
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import I
from sympy.core.relational import Eq, Equality
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Dummy, Symbol
from sympy.core.function import (expand_mul, expand, Derivative,
from sympy.functions import (exp, im, cos, sin, re, Piecewise,
from sympy.functions.combinatorial.factorials import factorial
from sympy.matrices import zeros, Matrix, NonSquareMatrixError, MatrixBase, eye
from sympy.polys import Poly, together
from sympy.simplify import collect, radsimp, signsimp # type: ignore
from sympy.simplify.powsimp import powdenest, powsimp
from sympy.simplify.ratsimp import ratsimp
from sympy.simplify.simplify import simplify
from sympy.sets.sets import FiniteSet
from sympy.solvers.deutils import ode_order
from sympy.solvers.solveset import NonlinearError, solveset
from sympy.utilities.iterables import (connected_components, iterable,
from sympy.utilities.misc import filldedent
from sympy.integrals.integrals import Integral, integrate
def simp_coeff_dep(expr, wrt1, wrt2=None):
    """Split rhs into terms, split terms into dep and coeff and collect on dep"""
    add_dep_terms = lambda e: e.is_Add and e.has(*wrt1)
    expandable = lambda e: e.is_Mul and any(map(add_dep_terms, e.args))
    expand_func = lambda e: expand_mul(e, deep=False)
    expand_mul_mod = lambda e: e.replace(expandable, expand_func)
    terms = Add.make_args(expand_mul_mod(expr))
    dc = {}
    for term in terms:
        coeff, dep = term.as_independent(*wrt1, as_Add=False)
        dep = simpdep(dep, wrt1)
        if dep is not S.One:
            dep2 = factor_terms(dep)
            if not dep2.has(*wrt1):
                coeff *= dep2
                dep = S.One
        if dep not in dc:
            dc[dep] = coeff
        else:
            dc[dep] += coeff
    termpairs = ((simpcoeff(c, wrt2), d) for d, c in dc.items())
    if wrt2 is not None:
        termpairs = ((simp_coeff_dep(c, wrt2), d) for c, d in termpairs)
    return Add(*(c * d for c, d in termpairs))