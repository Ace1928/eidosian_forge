from __future__ import annotations
from sympy.core import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, symbols as _symbols
from sympy.core.sympify import CantSympify
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.magic import pollute
from sympy.utilities.misc import as_int
def is_dependent(self, word):
    """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**4*y**-3).is_dependent(x**4*y**-2)
        True
        >>> (x**2*y**-1).is_dependent(x*y)
        False
        >>> (x*y**2*x*y**2).is_dependent(x*y**2)
        True
        >>> (x**12).is_dependent(x**-4)
        True

        See Also
        ========

        is_independent

        """
    try:
        return self.subword_index(word) is not None
    except ValueError:
        pass
    try:
        return self.subword_index(word ** (-1)) is not None
    except ValueError:
        return False