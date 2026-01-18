from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip, warns_deprecated_sympy
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import HAS_GMPY
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
import mpmath
from collections import defaultdict, OrderedDict
from mpmath.rational import mpq
def test_kernS():
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'
    assert -1 - 2 * (-(-x + 1 / x) / (x * (x - 1 / x) ** 2) - 1 / (x * (x - 1 / x))) == -1
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    s = '-1 - 2*(-(-x + 1/x)/(x*(x - 1/x)**2) - 1/(x*(x - 1/x)))'.replace('x', '_kern')
    ss = kernS(s)
    assert ss != -1 and ss.simplify() == -1
    assert kernS('Interval(-1,-2 - 4*(-3))') == Interval(-1, Add(-2, Mul(12, 1, evaluate=False), evaluate=False))
    assert kernS('_kern') == Symbol('_kern')
    assert kernS('E**-(x)') == exp(-x)
    e = 2 * (x + y) * y
    assert kernS(['2*(x + y)*y', ('2*(x + y)*y',)]) == [e, (e,)]
    assert kernS('-(2*sin(x)**2 + 2*sin(x)*cos(x))*y/2') == -y * (2 * sin(x) ** 2 + 2 * sin(x) * cos(x)) / 2
    assert kernS('(1 - x)/(1 - x*(1-y))') == kernS('(1-x)/(1-(1-y)*x)')
    assert kernS('(1-2**-(4+1)*(1-y)*x)') == 1 - x * (1 - y) / 32
    assert kernS('(1-2**(4+1)*(1-y)*x)') == 1 - 32 * x * (1 - y)
    assert kernS('(1-2.*(1-y)*x)') == 1 - 2.0 * x * (1 - y)
    one = kernS('x - (x - 1)')
    assert one != 1 and one.expand() == 1
    assert kernS('(2*x)/(x-1)') == 2 * x / (x - 1)