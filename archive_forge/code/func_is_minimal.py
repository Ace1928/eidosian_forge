from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query
def is_minimal(G, ring):
    """
    Checks if G is a minimal Groebner basis.
    """
    order = ring.order
    domain = ring.domain
    G.sort(key=lambda g: order(g.LM))
    for i, g in enumerate(G):
        if g.LC != domain.one:
            return False
        for h in G[:i] + G[i + 1:]:
            if monomial_divides(h.LM, g.LM):
                return False
    return True