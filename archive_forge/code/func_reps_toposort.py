from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def reps_toposort(r):
    """Sort replacements ``r`` so (k1, v1) appears before (k2, v2)
    if k2 is in v1's free symbols. This orders items in the
    way that cse returns its results (hence, in order to use the
    replacements in a substitution option it would make sense
    to reverse the order).

    Examples
    ========

    >>> from sympy.simplify.cse_main import reps_toposort
    >>> from sympy.abc import x, y
    >>> from sympy import Eq
    >>> for l, r in reps_toposort([(x, y + 1), (y, 2)]):
    ...     print(Eq(l, r))
    ...
    Eq(y, 2)
    Eq(x, y + 1)

    """
    r = sympify(r)
    E = []
    for c1, (k1, v1) in enumerate(r):
        for c2, (k2, v2) in enumerate(r):
            if k1 in v2.free_symbols:
                E.append((c1, c2))
    return [r[i] for i in topological_sort((range(len(r)), E))]