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
def test_issue_18531():
    M = Matrix([[1, 1, 1, 1, 1, 0, 1, 0, 0], [1 + sqrt(2), -1 + sqrt(2), 1 - sqrt(2), -sqrt(2) - 1, 1, 1, -1, 1, 1], [-5 + 2 * sqrt(2), -5 - 2 * sqrt(2), -5 - 2 * sqrt(2), -5 + 2 * sqrt(2), -7, 2, -7, -2, 0], [-3 * sqrt(2) - 1, 1 - 3 * sqrt(2), -1 + 3 * sqrt(2), 1 + 3 * sqrt(2), -7, -5, 7, -5, 3], [7 - 4 * sqrt(2), 4 * sqrt(2) + 7, 4 * sqrt(2) + 7, 7 - 4 * sqrt(2), 7, -12, 7, 12, 0], [-1 + 3 * sqrt(2), 1 + 3 * sqrt(2), -3 * sqrt(2) - 1, 1 - 3 * sqrt(2), 7, -5, -7, -5, 3], [-3 + 2 * sqrt(2), -3 - 2 * sqrt(2), -3 - 2 * sqrt(2), -3 + 2 * sqrt(2), -1, 2, -1, -2, 0], [1 - sqrt(2), -sqrt(2) - 1, 1 + sqrt(2), -1 + sqrt(2), -1, 1, 1, 1, 1]])
    with dotprodsimp(True):
        assert M.rref() == (Matrix([[1, 0, 0, 0, 0, 0, 0, 0, S(1) / 2], [0, 1, 0, 0, 0, 0, 0, 0, -S(1) / 2], [0, 0, 1, 0, 0, 0, 0, 0, S(1) / 2], [0, 0, 0, 1, 0, 0, 0, 0, -S(1) / 2], [0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, -S(1) / 2], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -S(1) / 2]]), (0, 1, 2, 3, 4, 5, 6, 7))