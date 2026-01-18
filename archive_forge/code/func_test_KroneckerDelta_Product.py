from sympy.concrete.products import (Product, product)
from sympy.concrete.summations import Sum
from sympy.core.function import (Derivative, Function, diff)
from sympy.core.numbers import (Rational, oo, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.combinatorial.factorials import (rf, factorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.simplify.combsimp import combsimp
from sympy.simplify.simplify import simplify
from sympy.testing.pytest import raises
def test_KroneckerDelta_Product():
    y = Symbol('y')
    assert Product(x * KroneckerDelta(x, y), (x, 0, 1)).doit() == 0