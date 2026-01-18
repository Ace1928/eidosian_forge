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
def test_mul_add_identity():
    m = Mul(1, 2)
    assert isinstance(m, Rational) and m.p == 2 and (m.q == 1)
    m = Mul(1, 2, evaluate=False)
    assert isinstance(m, Mul) and m.args == (1, 2)
    m = Mul(0, 1)
    assert m is S.Zero
    m = Mul(0, 1, evaluate=False)
    assert isinstance(m, Mul) and m.args == (0, 1)
    m = Add(0, 1)
    assert m is S.One
    m = Add(0, 1, evaluate=False)
    assert isinstance(m, Add) and m.args == (0, 1)