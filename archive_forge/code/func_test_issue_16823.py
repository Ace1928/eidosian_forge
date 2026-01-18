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
def test_issue_16823():
    M = Matrix(S('[\n        [1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I,15/128-3/32*I,19/256+551/1024*I],\n        [21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I,129/256-549/512*I,42533/16384+29103/8192*I],\n        [-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I,3/64+13/64*I,-23/32-59/256*I],\n        [1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I,119/128+143/128*I,-10879/2048+4343/4096*I],\n        [-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I,-9/32-1/16*I,183/256-97/128*I],\n        [1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I,-219/128+115/256*I,6301/4096-6609/1024*I],\n        [-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I,1/4-5/16*I,65/128+87/64*I],\n        [-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I,85/256-33/16*I,805/128+2415/512*I],\n        [0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I,1/4+1/2*I,-129/64-9/64*I],\n        [1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I,125/64+87/64*I,-2063/256+541/128*I],\n        [0,-4*I,0,-6,-4,9-5*I,-4*I,27/2+6*I,-2,17/4-13/2*I,1+I,-19/4+5/4*I,1/2-I,9/4+55/16*I,-3/4,45/32-37/16*I],\n        [0,1/4+1/2*I,1,-9/4+3*I,-2*I,119/8+29/4*I,1/4+5/2*I,-23/8-57/16*I,1/4+13/4*I,-825/64-147/32*I,21/8+I,-537/64+143/16*I,-5/8-39/16*I,2473/256+137/64*I,-149/64+49/32*I,-177/128-1369/128*I]]'))
    with dotprodsimp(True):
        assert M.rank() == 8