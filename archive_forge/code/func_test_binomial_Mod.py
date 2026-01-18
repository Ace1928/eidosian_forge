from sympy.concrete.products import Product
from sympy.core.function import expand_func
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core import EulerGamma
from sympy.core.numbers import (Float, I, Rational, nan, oo, pi, zoo)
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (ff, rf, binomial, factorial, factorial2)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import (gamma, polygamma)
from sympy.polys.polytools import Poly
from sympy.series.order import O
from sympy.simplify.simplify import simplify
from sympy.core.expr import unchanged
from sympy.core.function import ArgumentIndexError
from sympy.functions.combinatorial.factorials import subfactorial
from sympy.functions.special.gamma_functions import uppergamma
from sympy.testing.pytest import XFAIL, raises, slow
def test_binomial_Mod():
    p, q = (10 ** 5 + 3, 10 ** 9 + 33)
    r = 10 ** 7 + 5
    assert Mod(binomial(156675, 4433, evaluate=False), p) == Mod(binomial(156675, 4433), p)
    assert Mod(binomial(1234, 432, evaluate=False), q) == Mod(binomial(1234, 432), q)
    assert Mod(binomial(253, 113, evaluate=False), r) == Mod(binomial(253, 113), r)