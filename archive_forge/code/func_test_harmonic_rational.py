import string
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.function import (diff, expand_func)
from sympy.core import (EulerGamma, TribonacciConstant)
from sympy.core.numbers import (Float, I, Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.numbers import carmichael
from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.integers import floor
from sympy.polys.polytools import cancel
from sympy.series.limits import limit, Limit
from sympy.series.order import O
from sympy.functions import (
from sympy.functions.combinatorial.numbers import _nT
from sympy.core.expr import unchanged
from sympy.core.numbers import GoldenRatio, Integer
from sympy.testing.pytest import raises, nocache_fail, warns_deprecated_sympy
from sympy.abc import x
def test_harmonic_rational():
    ne = S(6)
    no = S(5)
    pe = S(8)
    po = S(9)
    qe = S(10)
    qo = S(13)
    Heee = harmonic(ne + pe / qe)
    Aeee = -log(10) + 2 * (Rational(-1, 4) + sqrt(5) / 4) * log(sqrt(-sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 - Rational(1, 4)) * log(sqrt(sqrt(5) / 8 + Rational(5, 8))) + pi * sqrt(2 * sqrt(5) / 5 + 1) / 2 + Rational(13944145, 4720968)
    Heeo = harmonic(ne + pe / qo)
    Aeeo = -log(26) + 2 * log(sin(pi * Rational(3, 13))) * cos(pi * Rational(4, 13)) + 2 * log(sin(pi * Rational(2, 13))) * cos(pi * Rational(32, 13)) + 2 * log(sin(pi * Rational(5, 13))) * cos(pi * Rational(80, 13)) - 2 * log(sin(pi * Rational(6, 13))) * cos(pi * Rational(5, 13)) - 2 * log(sin(pi * Rational(4, 13))) * cos(pi / 13) + pi * cot(pi * Rational(5, 13)) / 2 - 2 * log(sin(pi / 13)) * cos(pi * Rational(3, 13)) + Rational(2422020029, 702257080)
    Heoe = harmonic(ne + po / qe)
    Aeoe = -log(20) + 2 * (Rational(1, 4) + sqrt(5) / 4) * log(Rational(-1, 4) + sqrt(5) / 4) + 2 * (Rational(-1, 4) + sqrt(5) / 4) * log(sqrt(-sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 - Rational(1, 4)) * log(sqrt(sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 + Rational(1, 4)) * log(Rational(1, 4) + sqrt(5) / 4) + Rational(11818877030, 4286604231) + pi * sqrt(2 * sqrt(5) + 5) / 2
    Heoo = harmonic(ne + po / qo)
    Aeoo = -log(26) + 2 * log(sin(pi * Rational(3, 13))) * cos(pi * Rational(54, 13)) + 2 * log(sin(pi * Rational(4, 13))) * cos(pi * Rational(6, 13)) + 2 * log(sin(pi * Rational(6, 13))) * cos(pi * Rational(108, 13)) - 2 * log(sin(pi * Rational(5, 13))) * cos(pi / 13) - 2 * log(sin(pi / 13)) * cos(pi * Rational(5, 13)) + pi * cot(pi * Rational(4, 13)) / 2 - 2 * log(sin(pi * Rational(2, 13))) * cos(pi * Rational(3, 13)) + Rational(11669332571, 3628714320)
    Hoee = harmonic(no + pe / qe)
    Aoee = -log(10) + 2 * (Rational(-1, 4) + sqrt(5) / 4) * log(sqrt(-sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 - Rational(1, 4)) * log(sqrt(sqrt(5) / 8 + Rational(5, 8))) + pi * sqrt(2 * sqrt(5) / 5 + 1) / 2 + Rational(779405, 277704)
    Hoeo = harmonic(no + pe / qo)
    Aoeo = -log(26) + 2 * log(sin(pi * Rational(3, 13))) * cos(pi * Rational(4, 13)) + 2 * log(sin(pi * Rational(2, 13))) * cos(pi * Rational(32, 13)) + 2 * log(sin(pi * Rational(5, 13))) * cos(pi * Rational(80, 13)) - 2 * log(sin(pi * Rational(6, 13))) * cos(pi * Rational(5, 13)) - 2 * log(sin(pi * Rational(4, 13))) * cos(pi / 13) + pi * cot(pi * Rational(5, 13)) / 2 - 2 * log(sin(pi / 13)) * cos(pi * Rational(3, 13)) + Rational(53857323, 16331560)
    Hooe = harmonic(no + po / qe)
    Aooe = -log(20) + 2 * (Rational(1, 4) + sqrt(5) / 4) * log(Rational(-1, 4) + sqrt(5) / 4) + 2 * (Rational(-1, 4) + sqrt(5) / 4) * log(sqrt(-sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 - Rational(1, 4)) * log(sqrt(sqrt(5) / 8 + Rational(5, 8))) + 2 * (-sqrt(5) / 4 + Rational(1, 4)) * log(Rational(1, 4) + sqrt(5) / 4) + Rational(486853480, 186374097) + pi * sqrt(2 * sqrt(5) + 5) / 2
    Hooo = harmonic(no + po / qo)
    Aooo = -log(26) + 2 * log(sin(pi * Rational(3, 13))) * cos(pi * Rational(54, 13)) + 2 * log(sin(pi * Rational(4, 13))) * cos(pi * Rational(6, 13)) + 2 * log(sin(pi * Rational(6, 13))) * cos(pi * Rational(108, 13)) - 2 * log(sin(pi * Rational(5, 13))) * cos(pi / 13) - 2 * log(sin(pi / 13)) * cos(pi * Rational(5, 13)) + pi * cot(pi * Rational(4, 13)) / 2 - 2 * log(sin(pi * Rational(2, 13))) * cos(3 * pi / 13) + Rational(383693479, 125128080)
    H = [Heee, Heeo, Heoe, Heoo, Hoee, Hoeo, Hooe, Hooo]
    A = [Aeee, Aeeo, Aeoe, Aeoo, Aoee, Aoeo, Aooe, Aooo]
    for h, a in zip(H, A):
        e = expand_func(h).doit()
        assert cancel(e / a) == 1
        assert abs(h.n() - a.n()) < 1e-12