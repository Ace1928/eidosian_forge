from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning
@property
def has_finite_limits(self):
    """
        Returns True if the limits are known to be finite, either by the
        explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be infinite, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 1, 8)).has_finite_limits
        True

        >>> Integral(x, (x, 1, oo)).has_finite_limits
        False

        >>> M = Symbol('M')
        >>> Sum(x, (x, 1, M)).has_finite_limits

        >>> N = Symbol('N', integer=True)
        >>> Product(x, (x, 1, N)).has_finite_limits
        True

        See Also
        ========

        has_reversed_limits

        """
    ret_None = False
    for lim in self.limits:
        if len(lim) == 3:
            if any((l.is_infinite for l in lim[1:])):
                return False
            elif any((l.is_infinite is None for l in lim[1:])):
                if lim[0].is_infinite is None:
                    ret_None = True
        elif lim[0].is_infinite is None:
            ret_None = True
    if ret_None:
        return None
    return True