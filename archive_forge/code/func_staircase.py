from itertools import combinations_with_replacement
from sympy.core import symbols, Add, Dummy
from sympy.core.numbers import Rational
from sympy.polys import cancel, ComputationFailed, parallel_poly_from_expr, reduced, Poly
from sympy.polys.monomials import Monomial, monomial_div
from sympy.polys.polyerrors import DomainError, PolificationFailed
from sympy.utilities.misc import debug, debugf
def staircase(n):
    """
        Compute all monomials with degree less than ``n`` that are
        not divisible by any element of ``leading_monomials``.
        """
    if n == 0:
        return [1]
    S = []
    for mi in combinations_with_replacement(range(len(opt.gens)), n):
        m = [0] * len(opt.gens)
        for i in mi:
            m[i] += 1
        if all((monomial_div(m, lmg) is None for lmg in leading_monomials)):
            S.append(m)
    return [Monomial(s).as_expr(*opt.gens) for s in S] + staircase(n - 1)