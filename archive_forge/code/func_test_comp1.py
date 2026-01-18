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
def test_comp1():
    a = sqrt(2).n(7)
    assert comp(a, 1.4142129) is False
    assert comp(a, 1.414213)
    assert comp(a, 1.4142141)
    assert comp(a, 1.4142142) is False
    assert comp(sqrt(2).n(2), '1.4')
    assert comp(sqrt(2).n(2), Float(1.4, 2), '')
    assert comp(sqrt(2).n(2), 1.4, '')
    assert comp(sqrt(2).n(2), Float(1.4, 3), '') is False
    assert comp(sqrt(2) + sqrt(3) * I, 1.4 + 1.7 * I, 0.1)
    assert not comp(sqrt(2) + sqrt(3) * I, (1.5 + 1.7 * I) * 0.89, 0.1)
    assert comp(sqrt(2) + sqrt(3) * I, (1.5 + 1.7 * I) * 0.9, 0.1)
    assert comp(sqrt(2) + sqrt(3) * I, (1.5 + 1.7 * I) * 1.07, 0.1)
    assert not comp(sqrt(2) + sqrt(3) * I, (1.5 + 1.7 * I) * 1.08, 0.1)
    assert [(i, j) for i in range(130, 150) for j in range(170, 180) if comp((sqrt(2) + I * sqrt(3)).n(3), i / 100.0 + I * j / 100.0)] == [(141, 173), (142, 173)]
    raises(ValueError, lambda: comp(t, '1'))
    raises(ValueError, lambda: comp(t, 1))
    assert comp(0, 0.0)
    assert comp(0.5, S.Half)
    assert comp(2 + sqrt(2), 2.0 + sqrt(2))
    assert not comp(0, 1)
    assert not comp(2, sqrt(2))
    assert not comp(2 + I, 2.0 + sqrt(2))
    assert not comp(2.0 + sqrt(2), 2 + I)
    assert not comp(2.0 + sqrt(2), sqrt(3))
    assert comp(1 / pi.n(4), 0.3183, 1e-05)
    assert not comp(1 / pi.n(4), 0.3183, 8e-06)