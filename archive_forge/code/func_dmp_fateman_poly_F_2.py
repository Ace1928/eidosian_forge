from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent
from sympy.polys.rings import ring
def dmp_fateman_poly_F_2(n, K):
    """Fateman's GCD benchmark: linearly dense quartic inputs """
    u = [K(1), K(0)]
    for i in range(n - 1):
        u = [dmp_one(i, K), u]
    m = n - 1
    v = dmp_add_term(u, dmp_ground(K(2), m - 1), 0, n, K)
    f = dmp_sqr([dmp_one(m, K), dmp_neg(v, m, K)], n, K)
    g = dmp_sqr([dmp_one(m, K), v], n, K)
    v = dmp_add_term(u, dmp_one(m - 1, K), 0, n, K)
    h = dmp_sqr([dmp_one(m, K), v], n, K)
    return (dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h)