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
def cyclic_reduction(self, removed=False):
    """Return a cyclically reduced version of the word. Unlike
        `identity_cyclic_reduction`, this will not cyclically permute
        the reduced word - just remove the "unreduced" bits on either
        side of it. Compare the examples with those of
        `identity_cyclic_reduction`.

        When `removed` is `True`, return a tuple `(word, r)` where
        self `r` is such that before the reduction the word was either
        `r*word*r**-1`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction()
        y**-1*x**2
        >>> (x**-3*y**-1*x**5).cyclic_reduction(removed=True)
        (y**-1*x**2, x**-3)

        """
    word = self.copy()
    g = self.group.identity
    while not word.is_cyclically_reduced():
        exp1 = abs(word.exponent_syllable(0))
        exp2 = abs(word.exponent_syllable(-1))
        exp = min(exp1, exp2)
        start = word[0] ** abs(exp)
        end = word[-1] ** abs(exp)
        word = start ** (-1) * word * end ** (-1)
        g = g * start
    if removed:
        return (word, g)
    return word