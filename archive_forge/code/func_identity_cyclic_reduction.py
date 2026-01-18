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
def identity_cyclic_reduction(self):
    """Return a unique cyclically reduced version of the word.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> (x**2*y**2*x**-1).identity_cyclic_reduction()
        x*y**2
        >>> (x**-3*y**-1*x**5).identity_cyclic_reduction()
        x**2*y**-1

        References
        ==========

        .. [1] https://planetmath.org/cyclicallyreduced

        """
    word = self.copy()
    group = self.group
    while not word.is_cyclically_reduced():
        exp1 = word.exponent_syllable(0)
        exp2 = word.exponent_syllable(-1)
        r = exp1 + exp2
        if r == 0:
            rep = word.array_form[1:word.number_syllables() - 1]
        else:
            rep = ((word.generator_syllable(0), exp1 + exp2),) + word.array_form[1:word.number_syllables() - 1]
        word = group.dtype(rep)
    return word