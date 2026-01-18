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
def test_issue_14706():
    if not numpy:
        skip('numpy not installed.')
    z1 = numpy.zeros((1, 1), dtype=numpy.float64)
    z2 = numpy.zeros((2, 2), dtype=numpy.float64)
    z3 = numpy.zeros((), dtype=numpy.float64)
    y1 = numpy.ones((1, 1), dtype=numpy.float64)
    y2 = numpy.ones((2, 2), dtype=numpy.float64)
    y3 = numpy.ones((), dtype=numpy.float64)
    assert numpy.all(x + z1 == numpy.full((1, 1), x))
    assert numpy.all(x + z2 == numpy.full((2, 2), x))
    assert numpy.all(z1 + x == numpy.full((1, 1), x))
    assert numpy.all(z2 + x == numpy.full((2, 2), x))
    for z in [z3, numpy.int64(0), numpy.float64(0), numpy.complex64(0)]:
        assert x + z == x
        assert z + x == x
        assert isinstance(x + z, Symbol)
        assert isinstance(z + x, Symbol)
    assert numpy.all(x + y1 == numpy.full((1, 1), x + 1.0))
    assert numpy.all(x + y2 == numpy.full((2, 2), x + 1.0))
    assert numpy.all(y1 + x == numpy.full((1, 1), x + 1.0))
    assert numpy.all(y2 + x == numpy.full((2, 2), x + 1.0))
    for y_ in [y3, numpy.int64(1), numpy.float64(1), numpy.complex64(1)]:
        assert x + y_ == y_ + x
        assert isinstance(x + y_, Add)
        assert isinstance(y_ + x, Add)
    assert x + numpy.array(x) == 2 * x
    assert x + numpy.array([x]) == numpy.array([2 * x], dtype=object)
    assert sympify(numpy.array([1])) == ImmutableDenseNDimArray([1], 1)
    assert sympify(numpy.array([[[1]]])) == ImmutableDenseNDimArray([1], (1, 1, 1))
    assert sympify(z1) == ImmutableDenseNDimArray([0.0], (1, 1))
    assert sympify(z2) == ImmutableDenseNDimArray([0.0, 0.0, 0.0, 0.0], (2, 2))
    assert sympify(z3) == ImmutableDenseNDimArray([0.0], ())
    assert sympify(z3, strict=True) == 0.0
    raises(SympifyError, lambda: sympify(numpy.array([1]), strict=True))
    raises(SympifyError, lambda: sympify(z1, strict=True))
    raises(SympifyError, lambda: sympify(z2, strict=True))