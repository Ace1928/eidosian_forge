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
def exponent_sum(self, gen):
    """
        For an associative word `self` and a generator or inverse of generator
        `gen`, ``exponent_sum`` returns the number of times `gen` appears in
        `self` minus the number of times its inverse appears in `self`. If
        neither `gen` nor its inverse occur in `self` then 0 is returned.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.exponent_sum(x)
        2
        >>> w.exponent_sum(x**-1)
        -2
        >>> w = x**2*y**4*x**-3
        >>> w.exponent_sum(x)
        -1

        See Also
        ========

        generator_count

        """
    if len(gen) != 1:
        raise ValueError('gen must be a generator or inverse of a generator')
    s = gen.array_form[0]
    return s[1] * sum([i[1] for i in self.array_form if i[0] == s[0]])