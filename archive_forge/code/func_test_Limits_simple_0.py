from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (asin, cos, sin, tan)
from sympy.polys.rationaltools import together
from sympy.series.limits import limit
def test_Limits_simple_0():
    assert limit((2 ** (x + 1) + 3 ** (x + 1)) / (2 ** x + 3 ** x), x, oo) == 3