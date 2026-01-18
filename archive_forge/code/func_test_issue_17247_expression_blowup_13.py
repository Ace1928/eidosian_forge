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
def test_issue_17247_expression_blowup_13():
    M = Matrix([[0, 1 - x, x + 1, 1 - x], [1 - x, x + 1, 0, x + 1], [0, 1 - x, x + 1, 1 - x], [0, 0, 1 - x, 0]])
    ev = M.eigenvects()
    assert ev[0] == (0, 2, [Matrix([0, -1, 0, 1])])
    assert ev[1][0] == x - sqrt(2) * (x - 1) + 1
    assert ev[1][1] == 1
    assert ev[1][2][0].expand(deep=False, numer=True) == Matrix([[(-x + sqrt(2) * (x - 1) - 1) / (x - 1)], [-4 * x / (x ** 2 - 2 * x + 1) + (x + 1) * (x - sqrt(2) * (x - 1) + 1) / (x ** 2 - 2 * x + 1)], [(-x + sqrt(2) * (x - 1) - 1) / (x - 1)], [1]])
    assert ev[2][0] == x + sqrt(2) * (x - 1) + 1
    assert ev[2][1] == 1
    assert ev[2][2][0].expand(deep=False, numer=True) == Matrix([[(-x - sqrt(2) * (x - 1) - 1) / (x - 1)], [-4 * x / (x ** 2 - 2 * x + 1) + (x + 1) * (x + sqrt(2) * (x - 1) + 1) / (x ** 2 - 2 * x + 1)], [(-x - sqrt(2) * (x - 1) - 1) / (x - 1)], [1]])