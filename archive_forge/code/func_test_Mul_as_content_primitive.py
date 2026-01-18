from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, comp, nan,
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import (im, re, sign)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import (Max, sqrt)
from sympy.functions.elementary.trigonometric import (atan, cos, sin)
from sympy.polys.polytools import Poly
from sympy.sets.sets import FiniteSet
from sympy.core.parameters import distribute, evaluate
from sympy.core.expr import unchanged
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, warns
from sympy.utilities.exceptions import SymPyDeprecationWarning
from sympy.core.random import verify_numerically
from sympy.functions.elementary.trigonometric import asin
from itertools import product
def test_Mul_as_content_primitive():
    assert (2 * x).as_content_primitive() == (2, x)
    assert (x * (2 + 2 * x)).as_content_primitive() == (2, x * (1 + x))
    assert (x * (2 + 2 * y) * (3 * x + 3) ** 2).as_content_primitive() == (18, x * (1 + y) * (x + 1) ** 2)
    assert ((2 + 2 * x) ** 2 * (3 + 6 * x) + S.Half).as_content_primitive() == (S.Half, 24 * (x + 1) ** 2 * (2 * x + 1) + 1)