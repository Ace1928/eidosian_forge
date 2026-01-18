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
def test_evaluate_false():
    cases = {'2 + 3': Add(2, 3, evaluate=False), '2**2 / 3': Mul(Pow(2, 2, evaluate=False), Pow(3, -1, evaluate=False), evaluate=False), '2 + 3 * 5': Add(2, Mul(3, 5, evaluate=False), evaluate=False), '2 - 3 * 5': Add(2, Mul(-1, Mul(3, 5, evaluate=False), evaluate=False), evaluate=False), '1 / 3': Mul(1, Pow(3, -1, evaluate=False), evaluate=False), 'True | False': Or(True, False, evaluate=False), '1 + 2 + 3 + 5*3 + integrate(x)': Add(1, 2, 3, Mul(5, 3, evaluate=False), x ** 2 / 2, evaluate=False), '2 * 4 * 6 + 8': Add(Mul(2, 4, 6, evaluate=False), 8, evaluate=False), '2 - 8 / 4': Add(2, Mul(-1, Mul(8, Pow(4, -1, evaluate=False), evaluate=False), evaluate=False), evaluate=False), '2 - 2**2': Add(2, Mul(-1, Pow(2, 2, evaluate=False), evaluate=False), evaluate=False)}
    for case, result in cases.items():
        assert sympify(case, evaluate=False) == result