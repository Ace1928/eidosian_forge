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
def test_matrix_inverse_mod():
    A = Matrix(2, 1, [1, 0])
    raises(NonSquareMatrixError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 0, 0, 0])
    raises(ValueError, lambda: A.inv_mod(2))
    A = Matrix(2, 2, [1, 2, 3, 4])
    Ai = Matrix(2, 2, [1, 1, 0, 1])
    assert A.inv_mod(3) == Ai
    A = Matrix(2, 2, [1, 0, 0, 1])
    assert A.inv_mod(2) == A
    A = Matrix(3, 3, [1, 2, 3, 4, 5, 6, 7, 8, 9])
    raises(ValueError, lambda: A.inv_mod(5))
    A = Matrix(3, 3, [5, 1, 3, 2, 6, 0, 2, 1, 1])
    Ai = Matrix(3, 3, [6, 8, 0, 1, 5, 6, 5, 6, 4])
    assert A.inv_mod(9) == Ai
    A = Matrix(3, 3, [1, 6, -3, 4, 1, -5, 3, -5, 5])
    Ai = Matrix(3, 3, [4, 3, 3, 1, 2, 5, 1, 5, 1])
    assert A.inv_mod(6) == Ai
    A = Matrix(3, 3, [1, 6, 1, 4, 1, 5, 3, 2, 5])
    Ai = Matrix(3, 3, [6, 0, 3, 6, 6, 4, 1, 6, 1])
    assert A.inv_mod(7) == Ai