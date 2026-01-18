from sympy import MatAdd
from sympy.algebras.quaternion import Quaternion
from sympy.assumptions.ask import Q
from sympy.calculus.accumulationbounds import AccumBounds
from sympy.combinatorics.partitions import Partition
from sympy.concrete.summations import (Sum, summation)
from sympy.core.add import Add
from sympy.core.containers import (Dict, Tuple)
from sympy.core.expr import UnevaluatedExpr, Expr
from sympy.core.function import (Derivative, Function, Lambda, Subs, WildFunction)
from sympy.core.mul import Mul
from sympy.core import (Catalan, EulerGamma, GoldenRatio, TribonacciConstant)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi, zoo)
from sympy.core.parameters import _exp_is_pow
from sympy.core.power import Pow
from sympy.core.relational import (Eq, Rel, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild, symbols)
from sympy.functions.combinatorial.factorials import (factorial, factorial2, subfactorial)
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.functions.special.delta_functions import Heaviside
from sympy.functions.special.zeta_functions import zeta
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import (Equivalent, false, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.slice import MatrixSlice
from sympy.matrices import SparseMatrix
from sympy.polys.polytools import factor
from sympy.series.limits import Limit
from sympy.series.order import O
from sympy.sets.sets import (Complement, FiniteSet, Interval, SymmetricDifference)
from sympy.external import import_module
from sympy.physics.control.lti import TransferFunction, Series, Parallel, \
from sympy.physics.units import second, joule
from sympy.polys import (Poly, rootof, RootSum, groebner, ring, field, ZZ, QQ,
from sympy.geometry import Point, Circle, Polygon, Ellipse, Triangle
from sympy.tensor import NDimArray
from sympy.tensor.array.expressions.array_expressions import ArraySymbol, ArrayElement
from sympy.testing.pytest import raises, warns_deprecated_sympy
from sympy.printing import sstr, sstrrepr, StrPrinter
from sympy.physics.quantum.trace import Tr
def test_Float():
    assert str(Float('1.23', dps=1 + 2)) == '1.23'
    assert str(Float('1.23456789', dps=1 + 8)) == '1.23456789'
    assert str(Float('1.234567890123456789', dps=1 + 18)) == '1.234567890123456789'
    assert str(pi.evalf(1 + 2)) == '3.14'
    assert str(pi.evalf(1 + 14)) == '3.14159265358979'
    assert str(pi.evalf(1 + 64)) == '3.1415926535897932384626433832795028841971693993751058209749445923'
    assert str(pi.round(-1)) == '0.0'
    assert str((pi ** 400 - (pi ** 400).round(1)).n(2)) == '-0.e+88'
    assert sstr(Float('100'), full_prec=False, min=-2, max=2) == '1.0e+2'
    assert sstr(Float('100'), full_prec=False, min=-2, max=3) == '100.0'
    assert sstr(Float('0.1'), full_prec=False, min=-2, max=3) == '0.1'
    assert sstr(Float('0.099'), min=-2, max=3) == '9.90000000000000e-2'