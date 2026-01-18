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
def test_issue_17247_expression_blowup_26():
    M = Matrix(S('[\n        [             -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,      1/4 - 5*I/16,      65/128 + 87*I/64,         -9/32 - I/16,      183/256 - 97*I/128],\n        [-149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,  85/256 - 33*I/16,  805/128 + 2415*I/512, -219/128 + 115*I/256, 6301/4096 - 6609*I/1024],\n        [          1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,         1/4 + I/2,      -129/64 - 9*I/64,         1/4 - 5*I/16,        65/128 + 87*I/64],\n        [   -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,  125/64 + 87*I/64, -2063/256 + 541*I/128,     85/256 - 33*I/16,    805/128 + 2415*I/512],\n        [            1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,              -3/4,       45/32 - 37*I/16,            1/4 + I/2,        -129/64 - 9*I/64],\n        [         21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64, -149/64 + 49*I/32, -177/128 - 1369*I/128,     125/64 + 87*I/64,   -2063/256 + 541*I/128],\n        [               -2,         17/4 - 13*I/2,             1 + I,         -19/4 + 5*I/4,           1/2 - I,         9/4 + 55*I/16,                 -3/4,         45/32 - 37*I/16],\n        [     1/4 + 13*I/4,    -825/64 - 147*I/32,          21/8 + I,    -537/64 + 143*I/16,    -5/8 - 39*I/16,   2473/256 + 137*I/64,    -149/64 + 49*I/32,   -177/128 - 1369*I/128]]'))
    with dotprodsimp(True):
        assert M.rank() == 4