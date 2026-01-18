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
def test_int_float():

    class F1_1:

        def __float__(self):
            return 1.1

    class F1_1b:
        """
        This class is still a float, even though it also implements __int__().
        """

        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

    class F1_1c:
        """
        This class is still a float, because it implements _sympy_()
        """

        def __float__(self):
            return 1.1

        def __int__(self):
            return 1

        def _sympy_(self):
            return Float(1.1)

    class I5:

        def __int__(self):
            return 5

    class I5b:
        """
        This class implements both __int__() and __float__(), so it will be
        treated as Float in SymPy. One could change this behavior, by using
        float(a) == int(a), but deciding that integer-valued floats represent
        exact numbers is arbitrary and often not correct, so we do not do it.
        If, in the future, we decide to do it anyway, the tests for I5b need to
        be changed.
        """

        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

    class I5c:
        """
        This class implements both __int__() and __float__(), but also
        a _sympy_() method, so it will be Integer.
        """

        def __float__(self):
            return 5.0

        def __int__(self):
            return 5

        def _sympy_(self):
            return Integer(5)
    i5 = I5()
    i5b = I5b()
    i5c = I5c()
    f1_1 = F1_1()
    f1_1b = F1_1b()
    f1_1c = F1_1c()
    assert sympify(i5) == 5
    assert isinstance(sympify(i5), Integer)
    assert sympify(i5b) == 5.0
    assert isinstance(sympify(i5b), Float)
    assert sympify(i5c) == 5
    assert isinstance(sympify(i5c), Integer)
    assert abs(sympify(f1_1) - 1.1) < 1e-05
    assert abs(sympify(f1_1b) - 1.1) < 1e-05
    assert abs(sympify(f1_1c) - 1.1) < 1e-05
    assert _sympify(i5) == 5
    assert isinstance(_sympify(i5), Integer)
    assert _sympify(i5b) == 5.0
    assert isinstance(_sympify(i5b), Float)
    assert _sympify(i5c) == 5
    assert isinstance(_sympify(i5c), Integer)
    assert abs(_sympify(f1_1) - 1.1) < 1e-05
    assert abs(_sympify(f1_1b) - 1.1) < 1e-05
    assert abs(_sympify(f1_1c) - 1.1) < 1e-05