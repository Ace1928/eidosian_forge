from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import (Dict, Tuple)
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational, oo)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.integrals.integrals import Integral
from sympy.series.order import O
from sympy.sets.sets import Interval
from sympy.simplify.radsimp import collect
from sympy.simplify.simplify import simplify
from sympy.core.exprtools import (decompose_power, Factors, Term, _gcd_terms,
from sympy.core.mul import _keep_coeff as _keep_coeff
from sympy.simplify.cse_opts import sub_pre
from sympy.testing.pytest import raises
from sympy.abc import a, b, t, x, y, z
def test_Term():
    a = Term(4 * x * y ** 2 / z / t ** 3)
    b = Term(2 * x ** 3 * y ** 5 / t ** 3)
    assert a == Term(4, Factors({x: 1, y: 2}), Factors({z: 1, t: 3}))
    assert b == Term(2, Factors({x: 3, y: 5}), Factors({t: 3}))
    assert a.as_expr() == 4 * x * y ** 2 / z / t ** 3
    assert b.as_expr() == 2 * x ** 3 * y ** 5 / t ** 3
    assert a.inv() == Term(S.One / 4, Factors({z: 1, t: 3}), Factors({x: 1, y: 2}))
    assert b.inv() == Term(S.Half, Factors({t: 3}), Factors({x: 3, y: 5}))
    assert a.mul(b) == a * b == Term(8, Factors({x: 4, y: 7}), Factors({z: 1, t: 6}))
    assert a.quo(b) == a / b == Term(2, Factors({}), Factors({x: 2, y: 3, z: 1}))
    assert a.pow(3) == a ** 3 == Term(64, Factors({x: 3, y: 6}), Factors({z: 3, t: 9}))
    assert b.pow(3) == b ** 3 == Term(8, Factors({x: 9, y: 15}), Factors({t: 9}))
    assert a.pow(-3) == a ** (-3) == Term(S.One / 64, Factors({z: 3, t: 9}), Factors({x: 3, y: 6}))
    assert b.pow(-3) == b ** (-3) == Term(S.One / 8, Factors({t: 9}), Factors({x: 9, y: 15}))
    assert a.gcd(b) == Term(2, Factors({x: 1, y: 2}), Factors({t: 3}))
    assert a.lcm(b) == Term(4, Factors({x: 3, y: 5}), Factors({z: 1, t: 3}))
    a = Term(4 * x * y ** 2 / z / t ** 3)
    b = Term(2 * x ** 3 * y ** 5 * t ** 7)
    assert a.mul(b) == Term(8, Factors({x: 4, y: 7, t: 4}), Factors({z: 1}))
    assert Term((2 * x + 2) ** 3) == Term(8, Factors({x + 1: 3}), Factors({}))
    assert Term((2 * x + 2) * (3 * x + 6) ** 2) == Term(18, Factors({x + 1: 1, x + 2: 2}), Factors({}))