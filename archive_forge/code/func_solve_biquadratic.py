import itertools
from sympy.core import S
from sympy.core.sorting import default_sort_key
from sympy.polys import Poly, groebner, roots
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.polys.polyerrors import (ComputationFailed,
from sympy.simplify import rcollect
from sympy.utilities import postfixes
from sympy.utilities.misc import filldedent
def solve_biquadratic(f, g, opt):
    """Solve a system of two bivariate quadratic polynomial equations.

    Parameters
    ==========

    f: a single Expr or Poly
        First equation
    g: a single Expr or Poly
        Second Equation
    opt: an Options object
        For specifying keyword arguments and generators

    Returns
    =======

    List[Tuple]
        a list of tuples with elements being solutions for the
        symbols in the order they were passed as gens
    None
        None is returned when the computed basis contains only the ground.

    Examples
    ========

    >>> from sympy import Options, Poly
    >>> from sympy.abc import x, y
    >>> from sympy.solvers.polysys import solve_biquadratic
    >>> NewOption = Options((x, y), {'domain': 'ZZ'})

    >>> a = Poly(y**2 - 4 + x, y, x, domain='ZZ')
    >>> b = Poly(y*2 + 3*x - 7, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(1/3, 3), (41/27, 11/9)]

    >>> a = Poly(y + x**2 - 3, y, x, domain='ZZ')
    >>> b = Poly(-y + x - 4, y, x, domain='ZZ')
    >>> solve_biquadratic(a, b, NewOption)
    [(7/2 - sqrt(29)/2, -sqrt(29)/2 - 1/2), (sqrt(29)/2 + 7/2, -1/2 +       sqrt(29)/2)]
    """
    G = groebner([f, g])
    if len(G) == 1 and G[0].is_ground:
        return None
    if len(G) != 2:
        raise SolveFailed
    x, y = opt.gens
    p, q = G
    if not p.gcd(q).is_ground:
        raise SolveFailed
    p = Poly(p, x, expand=False)
    p_roots = [rcollect(expr, y) for expr in roots(p).keys()]
    q = q.ltrim(-1)
    q_roots = list(roots(q).keys())
    solutions = [(p_root.subs(y, q_root), q_root) for q_root, p_root in itertools.product(q_roots, p_roots)]
    return sorted(solutions, key=default_sort_key)