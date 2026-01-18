import random
import concurrent.futures
from collections.abc import Hashable
from sympy.core.add import Add
from sympy.core.function import (Function, diff, expand)
from sympy.core.numbers import (E, Float, I, Integer, Rational, nan, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.miscellaneous import (Max, Min, sqrt)
from sympy.functions.elementary.trigonometric import (cos, sin, tan)
from sympy.integrals.integrals import integrate
from sympy.polys.polytools import (Poly, PurePoly)
from sympy.printing.str import sstr
from sympy.sets.sets import FiniteSet
from sympy.simplify.simplify import (signsimp, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.matrices.matrices import (ShapeError, MatrixError,
from sympy.matrices import (
from sympy.matrices.utilities import _dotprodsimp_state
from sympy.core import Tuple, Wild
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.utilities.iterables import flatten, capture, iterable
from sympy.utilities.exceptions import ignore_warnings, SymPyDeprecationWarning
from sympy.testing.pytest import (raises, XFAIL, slow, skip, skip_under_pyodide,
from sympy.assumptions import Q
from sympy.tensor.array import Array
from sympy.matrices.expressions import MatPow
from sympy.algebras import Quaternion
from sympy.abc import a, b, c, d, x, y, z, t
def test_issue_17247_expression_blowup_21():
    M = Matrix(S('[\n        [             -3/4,       45/32 - 37*I/16,                   0,                     0],\n        [-149/64 + 49*I/32, -177/128 - 1369*I/128,                   0, -2063/256 + 541*I/128],\n        [                0,         9/4 + 55*I/16, 2473/256 + 137*I/64,                     0],\n        [                0,                     0,                   0, -177/128 - 1369*I/128]]'))
    with dotprodsimp(True):
        assert M.inv(method='GE') == Matrix(S('[\n            [-26194832/3470993 - 31733264*I/3470993, 156352/3470993 + 10325632*I/3470993, 0, -7741283181072/3306971225785 + 2999007604624*I/3306971225785],\n            [4408224/3470993 - 9675328*I/3470993, -2422272/3470993 + 1523712*I/3470993, 0, -1824666489984/3306971225785 - 1401091949952*I/3306971225785],\n            [-26406945676288/22270005630769 + 10245925485056*I/22270005630769, 7453523312640/22270005630769 + 1601616519168*I/22270005630769, 633088/6416033 - 140288*I/6416033, 872209227109521408/21217636514687010905 + 6066405081802389504*I/21217636514687010905],\n            [0, 0, 0, -11328/952745 + 87616*I/952745]]'))