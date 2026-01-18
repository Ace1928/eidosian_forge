import numbers as nums
import decimal
from sympy.concrete.summations import Sum
from sympy.core import (EulerGamma, Catalan, TribonacciConstant,
from sympy.core.containers import Tuple
from sympy.core.expr import unchanged
from sympy.core.logic import fuzzy_not
from sympy.core.mul import Mul
from sympy.core.numbers import (mpf_norm, mod_inverse, igcd, seterr,
from sympy.core.power import Pow
from sympy.core.relational import Ge, Gt, Le, Lt
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.integers import floor
from sympy.functions.combinatorial.numbers import fibonacci
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt, cbrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.polys.domains.realfield import RealField
from sympy.printing.latex import latex
from sympy.printing.repr import srepr
from sympy.simplify import simplify
from sympy.core.power import integer_nthroot, isqrt, integer_log
from sympy.polys.domains.groundtypes import PythonRational
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.utilities.iterables import permutations
from sympy.testing.pytest import XFAIL, raises, _both_exp_pow
from mpmath import mpf
from mpmath.rational import mpq
import mpmath
from sympy.core import numbers
@_both_exp_pow
def test_Infinity_inequations():
    assert oo > pi
    assert not oo < pi
    assert exp(-3) < oo
    assert _inf > pi
    assert not _inf < pi
    assert exp(-3) < _inf
    raises(TypeError, lambda: oo < I)
    raises(TypeError, lambda: oo <= I)
    raises(TypeError, lambda: oo > I)
    raises(TypeError, lambda: oo >= I)
    raises(TypeError, lambda: -oo < I)
    raises(TypeError, lambda: -oo <= I)
    raises(TypeError, lambda: -oo > I)
    raises(TypeError, lambda: -oo >= I)
    raises(TypeError, lambda: I < oo)
    raises(TypeError, lambda: I <= oo)
    raises(TypeError, lambda: I > oo)
    raises(TypeError, lambda: I >= oo)
    raises(TypeError, lambda: I < -oo)
    raises(TypeError, lambda: I <= -oo)
    raises(TypeError, lambda: I > -oo)
    raises(TypeError, lambda: I >= -oo)
    assert oo > -oo and oo >= -oo
    assert (oo < -oo) == False and (oo <= -oo) == False
    assert -oo < oo and -oo <= oo
    assert (-oo > oo) == False and (-oo >= oo) == False
    assert (oo < oo) == False
    assert (oo > oo) == False
    assert (-oo > -oo) == False and (-oo < -oo) == False
    assert oo >= oo and oo <= oo and (-oo >= -oo) and (-oo <= -oo)
    assert (-oo < -_inf) == False
    assert (oo > _inf) == False
    assert -oo >= -_inf
    assert oo <= _inf
    x = Symbol('x')
    b = Symbol('b', finite=True, real=True)
    assert (x < oo) == Lt(x, oo)
    assert b < oo and b > -oo and (b <= oo) and (b >= -oo)
    assert oo > b and oo >= b and ((oo < b) == False) and ((oo <= b) == False)
    assert (-oo > b) == False and (-oo >= b) == False and (-oo < b) and (-oo <= b)
    assert (oo < x) == Lt(oo, x) and (oo > x) == Gt(oo, x)
    assert (oo <= x) == Le(oo, x) and (oo >= x) == Ge(oo, x)
    assert (-oo < x) == Lt(-oo, x) and (-oo > x) == Gt(-oo, x)
    assert (-oo <= x) == Le(-oo, x) and (-oo >= x) == Ge(-oo, x)