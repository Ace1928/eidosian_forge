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
def test_Poly():
    assert str(Poly(0, x)) == "Poly(0, x, domain='ZZ')"
    assert str(Poly(1, x)) == "Poly(1, x, domain='ZZ')"
    assert str(Poly(x, x)) == "Poly(x, x, domain='ZZ')"
    assert str(Poly(2 * x + 1, x)) == "Poly(2*x + 1, x, domain='ZZ')"
    assert str(Poly(2 * x - 1, x)) == "Poly(2*x - 1, x, domain='ZZ')"
    assert str(Poly(-1, x)) == "Poly(-1, x, domain='ZZ')"
    assert str(Poly(-x, x)) == "Poly(-x, x, domain='ZZ')"
    assert str(Poly(-2 * x + 1, x)) == "Poly(-2*x + 1, x, domain='ZZ')"
    assert str(Poly(-2 * x - 1, x)) == "Poly(-2*x - 1, x, domain='ZZ')"
    assert str(Poly(x - 1, x)) == "Poly(x - 1, x, domain='ZZ')"
    assert str(Poly(2 * x + x ** 5, x)) == "Poly(x**5 + 2*x, x, domain='ZZ')"
    assert str(Poly(3 ** (2 * x), 3 ** x)) == "Poly((3**x)**2, 3**x, domain='ZZ')"
    assert str(Poly((x ** 2) ** x)) == "Poly(((x**2)**x), (x**2)**x, domain='ZZ')"
    assert str(Poly((x + y) ** 3, x + y, expand=False)) == "Poly((x + y)**3, x + y, domain='ZZ')"
    assert str(Poly((x - 1) ** 2, x - 1, expand=False)) == "Poly((x - 1)**2, x - 1, domain='ZZ')"
    assert str(Poly(x ** 2 + 1 + y, x)) == "Poly(x**2 + y + 1, x, domain='ZZ[y]')"
    assert str(Poly(x ** 2 - 1 + y, x)) == "Poly(x**2 + y - 1, x, domain='ZZ[y]')"
    assert str(Poly(x ** 2 + I * x, x)) == "Poly(x**2 + I*x, x, domain='ZZ_I')"
    assert str(Poly(x ** 2 - I * x, x)) == "Poly(x**2 - I*x, x, domain='ZZ_I')"
    assert str(Poly(-x * y * z + x * y - 1, x, y, z)) == "Poly(-x*y*z + x*y - 1, x, y, z, domain='ZZ')"
    assert str(Poly(-w * x ** 21 * y ** 7 * z + (1 + w) * z ** 3 - 2 * x * z + 1, x, y, z)) == "Poly(-w*x**21*y**7*z - 2*x*z + (w + 1)*z**3 + 1, x, y, z, domain='ZZ[w]')"
    assert str(Poly(x ** 2 + 1, x, modulus=2)) == 'Poly(x**2 + 1, x, modulus=2)'
    assert str(Poly(2 * x ** 2 + 3 * x + 4, x, modulus=17)) == 'Poly(2*x**2 + 3*x + 4, x, modulus=17)'