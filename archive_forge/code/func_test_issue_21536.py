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
def test_issue_21536():
    u = sympify('x+3*x+2', evaluate=False)
    v = sympify('2*x+4*x+2+4', evaluate=False)
    assert u.is_Add and set(u.args) == {x, 3 * x, 2}
    assert v.is_Add and set(v.args) == {2 * x, 4 * x, 2, 4}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4'], evaluate=False) == [u, v]
    u = sympify('x+3*x+2', evaluate=True)
    v = sympify('2*x+4*x+2+4', evaluate=True)
    assert u.is_Add and set(u.args) == {4 * x, 2}
    assert v.is_Add and set(v.args) == {6 * x, 6}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4'], evaluate=True) == [u, v]
    u = sympify('x+3*x+2')
    v = sympify('2*x+4*x+2+4')
    assert u.is_Add and set(u.args) == {4 * x, 2}
    assert v.is_Add and set(v.args) == {6 * x, 6}
    assert sympify(['x+3*x+2', '2*x+4*x+2+4']) == [u, v]