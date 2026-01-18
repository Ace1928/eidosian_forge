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
def test_inv_block():
    a = Matrix([[1, 2], [2, 3]])
    b = Matrix([[3, x], [y, 3]])
    c = Matrix([[3, x, 3], [y, 3, z], [x, y, z]])
    A = diag(a, b, b)
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), b.inv())
    A = diag(a, b, c)
    assert A.inv(try_block_diag=True) == diag(a.inv(), b.inv(), c.inv())
    A = diag(a, c, b)
    assert A.inv(try_block_diag=True) == diag(a.inv(), c.inv(), b.inv())
    A = diag(a, a, b, a, c, a)
    assert A.inv(try_block_diag=True) == diag(a.inv(), a.inv(), b.inv(), a.inv(), c.inv(), a.inv())
    assert A.inv(try_block_diag=True, method='ADJ') == diag(a.inv(method='ADJ'), a.inv(method='ADJ'), b.inv(method='ADJ'), a.inv(method='ADJ'), c.inv(method='ADJ'), a.inv(method='ADJ'))