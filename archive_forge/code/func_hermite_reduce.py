from the names used in Bronstein's book.
from types import GeneratorType
from functools import reduce
from sympy.core.function import Lambda
from sympy.core.mul import Mul
from sympy.core.numbers import ilcm, I, oo
from sympy.core.power import Pow
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.elementary.exponential import log, exp
from sympy.functions.elementary.hyperbolic import (cosh, coth, sinh,
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (atan, sin, cos,
from .integrals import integrate, Integral
from .heurisch import _symbols
from sympy.polys.polyerrors import DomainError, PolynomialError
from sympy.polys.polytools import (real_roots, cancel, Poly, gcd,
from sympy.polys.rootoftools import RootSum
from sympy.utilities.iterables import numbered_symbols
def hermite_reduce(a, d, DE):
    """
    Hermite Reduction - Mack's Linear Version.

    Given a derivation D on k(t) and f = a/d in k(t), returns g, h, r in
    k(t) such that f = Dg + h + r, h is simple, and r is reduced.

    """
    l = Poly(1 / d.LC(), DE.t)
    a, d = (a.mul(l), d.mul(l))
    fp, fs, fn = canonical_representation(a, d, DE)
    a, d = fn
    l = Poly(1 / d.LC(), DE.t)
    a, d = (a.mul(l), d.mul(l))
    ga = Poly(0, DE.t)
    gd = Poly(1, DE.t)
    dd = derivation(d, DE)
    dm = gcd(d.to_field(), dd.to_field()).as_poly(DE.t)
    ds, _ = d.div(dm)
    while dm.degree(DE.t) > 0:
        ddm = derivation(dm, DE)
        dm2 = gcd(dm.to_field(), ddm.to_field())
        dms, _ = dm.div(dm2)
        ds_ddm = ds.mul(ddm)
        ds_ddm_dm, _ = ds_ddm.div(dm)
        b, c = gcdex_diophantine(-ds_ddm_dm.as_poly(DE.t), dms.as_poly(DE.t), a.as_poly(DE.t))
        b, c = (b.as_poly(DE.t), c.as_poly(DE.t))
        db = derivation(b, DE).as_poly(DE.t)
        ds_dms, _ = ds.div(dms)
        a = c.as_poly(DE.t) - db.mul(ds_dms).as_poly(DE.t)
        ga = ga * dm + b * gd
        gd = gd * dm
        ga, gd = ga.cancel(gd, include=True)
        dm = dm2
    q, r = a.div(ds)
    ga, gd = ga.cancel(gd, include=True)
    r, d = r.cancel(ds, include=True)
    rra = q * fs[1] + fp * fs[1] + fs[0]
    rrd = fs[1]
    rra, rrd = rra.cancel(rrd, include=True)
    return ((ga, gd), (r, d), (rra, rrd))