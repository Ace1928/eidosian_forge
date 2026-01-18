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
def sub_syllables(self, from_i, to_j):
    """
        `sub_syllables` returns the subword of the associative word `self` that
        consists of syllables from positions `from_to` to `to_j`, where
        `from_to` and `to_j` must be positive integers and indexing is done
        with origin 0.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> f, a, b = free_group("a, b")
        >>> w = a**5*b*a**2*b**-4*a
        >>> w.sub_syllables(1, 2)
        b
        >>> w.sub_syllables(3, 3)
        <identity>

        """
    if not isinstance(from_i, int) or not isinstance(to_j, int):
        raise ValueError('both arguments should be integers')
    group = self.group
    if to_j <= from_i:
        return group.identity
    else:
        r = tuple(self.array_form[from_i:to_j])
        return group.dtype(r)