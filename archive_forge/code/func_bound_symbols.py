import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
@property
def bound_symbols(self):
    """The list of dummy variables.

        Note
        ====

        Note that all variables are dummy variables since a limit without
        lower bound or upper bound is not accepted.
        """
    return [l[0] for l in self._limits if len(l) != 1]