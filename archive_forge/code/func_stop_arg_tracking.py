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
def stop_arg_tracking(self, func_i):
    """
        Remove the function func_i from the argument to function mapping.
        """
    for arg in self.func_to_argset[func_i]:
        self.arg_to_funcset[arg].remove(func_i)