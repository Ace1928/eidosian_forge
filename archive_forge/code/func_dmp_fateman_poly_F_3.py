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
def dmp_fateman_poly_F_3(n, K):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    u = dup_from_raw_dict({n + 1: K.one}, K)
    for i in range(0, n - 1):
        u = dmp_add_term([u], dmp_one(i, K), n + 1, i + 1, K)
    v = dmp_add_term(u, dmp_ground(K(2), n - 2), 0, n, K)
    f = dmp_sqr(dmp_add_term([dmp_neg(v, n - 1, K)], dmp_one(n - 1, K), n + 1, n, K), n, K)
    g = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)
    v = dmp_add_term(u, dmp_one(n - 2, K), 0, n - 1, K)
    h = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)
    return (dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h)