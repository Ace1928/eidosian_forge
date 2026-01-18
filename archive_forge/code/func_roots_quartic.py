import math
from functools import reduce
from sympy.core import S, I, pi
from sympy.core.exprtools import factor_terms
from sympy.core.function import _mexpand
from sympy.core.logic import fuzzy_not
from sympy.core.mul import expand_2arg, Mul
from sympy.core.numbers import Rational, igcd, comp
from sympy.core.power import Pow
from sympy.core.relational import Eq
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.sympify import sympify
from sympy.functions import exp, im, cos, acos, Piecewise
from sympy.functions.elementary.miscellaneous import root, sqrt
from sympy.ntheory import divisors, isprime, nextprime
from sympy.polys.domains import EX
from sympy.polys.polyerrors import (PolynomialError, GeneratorsNeeded,
from sympy.polys.polyquinticconst import PolyQuintic
from sympy.polys.polytools import Poly, cancel, factor, gcd_list, discriminant
from sympy.polys.rationaltools import together
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.utilities import public
from sympy.utilities.misc import filldedent
def roots_quartic(f):
    """
    Returns a list of roots of a quartic polynomial.

    There are many references for solving quartic expressions available [1-5].
    This reviewer has found that many of them require one to select from among
    2 or more possible sets of solutions and that some solutions work when one
    is searching for real roots but do not work when searching for complex roots
    (though this is not always stated clearly). The following routine has been
    tested and found to be correct for 0, 2 or 4 complex roots.

    The quasisymmetric case solution [6] looks for quartics that have the form
    `x**4 + A*x**3 + B*x**2 + C*x + D = 0` where `(C/A)**2 = D`.

    Although no general solution that is always applicable for all
    coefficients is known to this reviewer, certain conditions are tested
    to determine the simplest 4 expressions that can be returned:

      1) `f = c + a*(a**2/8 - b/2) == 0`
      2) `g = d - a*(a*(3*a**2/256 - b/16) + c/4) = 0`
      3) if `f != 0` and `g != 0` and `p = -d + a*c/4 - b**2/12` then
        a) `p == 0`
        b) `p != 0`

    Examples
    ========

        >>> from sympy import Poly
        >>> from sympy.polys.polyroots import roots_quartic

        >>> r = roots_quartic(Poly('x**4-6*x**3+17*x**2-26*x+20'))

        >>> # 4 complex roots: 1+-I*sqrt(3), 2+-I
        >>> sorted(str(tmp.evalf(n=2)) for tmp in r)
        ['1.0 + 1.7*I', '1.0 - 1.7*I', '2.0 + 1.0*I', '2.0 - 1.0*I']

    References
    ==========

    1. http://mathforum.org/dr.math/faq/faq.cubic.equations.html
    2. https://en.wikipedia.org/wiki/Quartic_function#Summary_of_Ferrari.27s_method
    3. https://planetmath.org/encyclopedia/GaloisTheoreticDerivationOfTheQuarticFormula.html
    4. https://people.bath.ac.uk/masjhd/JHD-CA.pdf
    5. http://www.albmath.org/files/Math_5713.pdf
    6. https://web.archive.org/web/20171002081448/http://www.statemaster.com/encyclopedia/Quartic-equation
    7. https://eqworld.ipmnet.ru/en/solutions/ae/ae0108.pdf
    """
    _, a, b, c, d = f.monic().all_coeffs()
    if not d:
        return [S.Zero] + roots([1, a, b, c], multiple=True)
    elif (c / a) ** 2 == d:
        x, m = (f.gen, c / a)
        g = Poly(x ** 2 + a * x + b - 2 * m, x)
        z1, z2 = roots_quadratic(g)
        h1 = Poly(x ** 2 - z1 * x + m, x)
        h2 = Poly(x ** 2 - z2 * x + m, x)
        r1 = roots_quadratic(h1)
        r2 = roots_quadratic(h2)
        return r1 + r2
    else:
        a2 = a ** 2
        e = b - 3 * a2 / 8
        f = _mexpand(c + a * (a2 / 8 - b / 2))
        aon4 = a / 4
        g = _mexpand(d - aon4 * (a * (3 * a2 / 64 - b / 4) + c))
        if f.is_zero:
            y1, y2 = [sqrt(tmp) for tmp in roots([1, e, g], multiple=True)]
            return [tmp - aon4 for tmp in [-y1, -y2, y1, y2]]
        if g.is_zero:
            y = [S.Zero] + roots([1, 0, e, f], multiple=True)
            return [tmp - aon4 for tmp in y]
        else:
            sols = _roots_quartic_euler(e, f, g, aon4)
            if sols:
                return sols
            p = -e ** 2 / 12 - g
            q = -e ** 3 / 108 + e * g / 3 - f ** 2 / 8
            TH = Rational(1, 3)

            def _ans(y):
                w = sqrt(e + 2 * y)
                arg1 = 3 * e + 2 * y
                arg2 = 2 * f / w
                ans = []
                for s in [-1, 1]:
                    root = sqrt(-(arg1 + s * arg2))
                    for t in [-1, 1]:
                        ans.append((s * w - t * root) / 2 - aon4)
                return ans
            p = _mexpand(p)
            y1 = e * Rational(-5, 6) - q ** TH
            if p.is_zero:
                return _ans(y1)
            root = sqrt(q ** 2 / 4 + p ** 3 / 27)
            r = -q / 2 + root
            u = r ** TH
            y2 = e * Rational(-5, 6) + u - p / u / 3
            if fuzzy_not(p.is_zero):
                return _ans(y2)
            return [Piecewise((a1, Eq(p, 0)), (a2, True)) for a1, a2 in zip(_ans(y1), _ans(y2))]