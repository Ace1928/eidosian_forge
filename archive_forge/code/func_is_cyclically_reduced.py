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
def is_cyclically_reduced(self):
    """Returns whether the word is cyclically reduced or not.
        A word is cyclically reduced if by forming the cycle of the
        word, the word is not reduced, i.e a word w = `a_1 ... a_n`
        is called cyclically reduced if `a_1 \\ne a_n^{-1}`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**-1*x**-1).is_cyclically_reduced()
        False
        >>> (y*x**2*y**2).is_cyclically_reduced()
        True

        """
    if not self:
        return True
    return self[0] != self[-1] ** (-1)