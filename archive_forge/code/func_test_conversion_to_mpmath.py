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
@conserve_mpmath_dps
def test_conversion_to_mpmath():
    assert mpmath.mpmathify(Integer(1)) == mpmath.mpf(1)
    assert mpmath.mpmathify(S.Half) == mpmath.mpf(0.5)
    assert mpmath.mpmathify(Float('1.23', 15)) == mpmath.mpf('1.23')
    assert mpmath.mpmathify(I) == mpmath.mpc(1j)
    assert mpmath.mpmathify(1 + 2 * I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2 * I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1 + 2.0 * I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(1.0 + 2.0 * I) == mpmath.mpc(1 + 2j)
    assert mpmath.mpmathify(S.Half + S.Half * I) == mpmath.mpc(0.5 + 0.5j)
    assert mpmath.mpmathify(2 * I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(2.0 * I) == mpmath.mpc(2j)
    assert mpmath.mpmathify(S.Half * I) == mpmath.mpc(0.5j)
    mpmath.mp.dps = 100
    assert mpmath.mpmathify(pi.evalf(100) + pi.evalf(100) * I) == mpmath.pi + mpmath.pi * mpmath.j
    assert mpmath.mpmathify(pi.evalf(100) * I) == mpmath.pi * mpmath.j