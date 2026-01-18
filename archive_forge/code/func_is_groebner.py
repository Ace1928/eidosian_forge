from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def is_groebner(G, ring):
    """
    Check if G is a Groebner basis.
    """
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            s = spoly(G[i], G[j], ring)
            s = s.rem(G)
            if s:
                return False
    return True