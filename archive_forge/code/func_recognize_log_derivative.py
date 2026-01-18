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
def recognize_log_derivative(a, d, DE, z=None):
    """
    There exists a v in K(x)* such that f = dv/v
    where f a rational function if and only if f can be written as f = A/D
    where D is squarefree,deg(A) < deg(D), gcd(A, D) = 1,
    and all the roots of the Rothstein-Trager resultant are integers. In that case,
    any of the Rothstein-Trager, Lazard-Rioboo-Trager or Czichowski algorithm
    produces u in K(x) such that du/dx = uf.
    """
    z = z or Dummy('z')
    a, d = a.cancel(d, include=True)
    _, a = a.div(d)
    pz = Poly(z, DE.t)
    Dd = derivation(d, DE)
    q = a - pz * Dd
    r, _ = d.resultant(q, includePRS=True)
    r = Poly(r, z)
    Np, Sp = splitfactor_sqf(r, DE, coefficientD=True, z=z)
    for s, _ in Sp:
        a = real_roots(s.as_poly(z))
        if not all((j.is_Integer for j in a)):
            return False
    return True