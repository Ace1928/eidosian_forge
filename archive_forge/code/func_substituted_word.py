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
def substituted_word(self, from_i, to_j, by):
    """
        Returns the associative word obtained by replacing the subword of
        `self` that begins at position `from_i` and ends at position `to_j - 1`
        by the associative word `by`. `from_i` and `to_j` must be positive
        integers, indexing is done with origin 0. In other words,
        `w.substituted_word(w, from_i, to_j, by)` is the product of the three
        words: `w.subword(0, from_i)`, `by`, and
        `w.subword(to_j len(w))`.

        See Also
        ========

        eliminate_word

        """
    lw = len(self)
    if from_i >= to_j or from_i > lw or to_j > lw:
        raise ValueError('values should be within bounds')
    if from_i == 0 and to_j == lw:
        return by
    elif from_i == 0:
        return by * self.subword(to_j, lw)
    elif to_j == lw:
        return self.subword(0, from_i) * by
    else:
        return self.subword(0, from_i) * by * self.subword(to_j, lw)