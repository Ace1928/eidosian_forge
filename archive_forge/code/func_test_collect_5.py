from sympy.core.add import Add
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Rational)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, Wild, symbols)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (root, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys.polytools import factor
from sympy.series.order import O
from sympy.simplify.radsimp import (collect, collect_const, fraction, radsimp, rcollect)
from sympy.core.expr import unchanged
from sympy.core.mul import _unevaluated_Mul as umul
from sympy.simplify.radsimp import (_unevaluated_Add,
from sympy.testing.pytest import raises
from sympy.abc import x, y, z, a, b, c, d
def test_collect_5():
    """Collect with respect to a tuple"""
    a, x, y, z, n = symbols('a,x,y,z,n')
    assert collect(x ** 2 * y ** 4 + z * (x * y ** 2) ** 2 + z + a * z, [x * y ** 2, z]) in [z * (1 + a + x ** 2 * y ** 4) + x ** 2 * y ** 4, z * (1 + a) + x ** 2 * y ** 4 * (1 + z)]
    assert collect((1 + (x + y) + (x + y) ** 2).expand(), [x, y]) == 1 + y + x * (1 + 2 * y) + x ** 2 + y ** 2