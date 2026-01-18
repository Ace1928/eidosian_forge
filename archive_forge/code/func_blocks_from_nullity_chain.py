from types import FunctionType
from collections import Counter
from mpmath import mp, workprec
from mpmath.libmp.libmpf import prec_to_dps
from sympy.core.sorting import default_sort_key
from sympy.core.evalf import DEFAULT_MAXPREC, PrecisionExhausted
from sympy.core.logic import fuzzy_and, fuzzy_or
from sympy.core.numbers import Float
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.polys import roots, CRootOf, ZZ, QQ, EX
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.eigen import dom_eigenvects, dom_eigenvects_to_sympy
from sympy.polys.polytools import gcd
from .common import MatrixError, NonSquareMatrixError
from .determinant import _find_reasonable_pivot
from .utilities import _iszero, _simplify
def blocks_from_nullity_chain(d):
    """Return a list of the size of each Jordan block.
        If d_n is the nullity of E**n, then the number
        of Jordan blocks of size n is

            2*d_n - d_(n-1) - d_(n+1)"""
    mid = [2 * d[n] - d[n - 1] - d[n + 1] for n in range(1, len(d) - 1)]
    end = [d[-1] - d[-2]] if len(d) > 1 else [d[0]]
    return mid + end