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
def get_or_add_value_number(self, value):
    """
        Return the value number for the given argument.
        """
    nvalues = len(self.value_numbers)
    value_number = self.value_numbers.setdefault(value, nvalues)
    if value_number == nvalues:
        self.value_number_to_value.append(value)
        self.arg_to_funcset.append(OrderedSet())
    return value_number