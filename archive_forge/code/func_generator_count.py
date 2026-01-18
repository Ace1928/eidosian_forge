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
def generator_count(self, gen):
    """
        For an associative word `self` and a generator `gen`,
        ``generator_count`` returns the multiplicity of generator
        `gen` in `self`.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w = x**2*y**3
        >>> w.generator_count(x)
        2
        >>> w = x**2*y**4*x**-3
        >>> w.generator_count(x)
        5

        See Also
        ========

        exponent_sum

        """
    if len(gen) != 1 or gen.array_form[0][1] < 0:
        raise ValueError('gen must be a generator')
    s = gen.array_form[0]
    return s[1] * sum([abs(i[1]) for i in self.array_form if i[0] == s[0]])