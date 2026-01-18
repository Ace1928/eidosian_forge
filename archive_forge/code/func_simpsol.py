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
def simpsol(sol, wrt1, wrt2, doit=True):
    """Simplify solutions from dsolve_system."""

    def simprhs(rhs, rep, wrt1, wrt2):
        """Simplify the rhs of an ODE solution"""
        if rep:
            rhs = rhs.subs(rep)
        rhs = factor_terms(rhs)
        rhs = simp_coeff_dep(rhs, wrt1, wrt2)
        rhs = signsimp(rhs)
        return rhs

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

    def simpdep(term, wrt1):
        """Normalise factors involving t with powsimp and recombine exp"""

        def canonicalise(a):
            a = factor_terms(a)
            num, den = a.as_numer_denom()
            num = expand_mul(num)
            num = collect(num, wrt1)
            return num / den
        term = powsimp(term)
        rep = {e: exp(canonicalise(e.args[0])) for e in term.atoms(exp)}
        term = term.subs(rep)
        return term

    def simpcoeff(coeff, wrt2):
        """Bring to a common fraction and cancel with ratsimp"""
        coeff = together(coeff)
        if coeff.is_polynomial():
            coeff = ratsimp(radsimp(coeff))
        if wrt2 is not None:
            syms = list(wrt2) + list(ordered(coeff.free_symbols - set(wrt2)))
        else:
            syms = list(ordered(coeff.free_symbols))
        coeff = collect(coeff, syms)
        coeff = together(coeff)
        return coeff
    if doit:
        integrals = set().union(*(s.atoms(Integral) for s in sol))
        rep = {i: factor_terms(i).doit() for i in integrals}
    else:
        rep = {}
    sol = [Eq(s.lhs, simprhs(s.rhs, rep, wrt1, wrt2)) for s in sol]
    return sol