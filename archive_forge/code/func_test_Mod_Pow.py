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
def test_Mod_Pow():
    assert isinstance(Mod(Pow(2, 2, evaluate=False), 3), Integer)
    assert Mod(Pow(4, 13, evaluate=False), 497) == Mod(Pow(4, 13), 497)
    assert Mod(Pow(2, 10000000000, evaluate=False), 3) == 1
    assert Mod(Pow(32131231232, 9 ** 10 ** 6, evaluate=False), 10 ** 12) == pow(32131231232, 9 ** 10 ** 6, 10 ** 12)
    assert Mod(Pow(33284959323, 123 ** 999, evaluate=False), 11 ** 13) == pow(33284959323, 123 ** 999, 11 ** 13)
    assert Mod(Pow(78789849597, 333 ** 555, evaluate=False), 12 ** 9) == pow(78789849597, 333 ** 555, 12 ** 9)
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 16
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 6487
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 32191
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 18016
    expr = Pow(2, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 5137
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3 ** 10) == 16
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3 ** 10) == 256
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3 ** 10) == 6487
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3 ** 10) == 38281
    expr = Pow(expr, 2, evaluate=False)
    assert Mod(expr, 3 ** 10) == 15928
    expr = Pow(2, 2, evaluate=False)
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 256
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 9229
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 25708
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 26608
    expr = Pow(expr, expr, evaluate=False)
    assert Mod(expr, 3 ** 10) == 1966