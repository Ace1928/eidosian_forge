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
def get_subset_candidates(self, argset, restrict_to_funcset=None):
    """
        Return a set of functions each of which whose argument list contains
        ``argset``, optionally filtered only to contain functions in
        ``restrict_to_funcset``.
        """
    iarg = iter(argset)
    indices = OrderedSet((fi for fi in self.arg_to_funcset[next(iarg)]))
    if restrict_to_funcset is not None:
        indices &= restrict_to_funcset
    for arg in iarg:
        indices &= self.arg_to_funcset[arg]
    return indices