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
def contains_generators(self):
    """
        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y, z = free_group("x, y, z")
        >>> (x**2*y**-1).contains_generators()
        {x, y}
        >>> (x**3*z).contains_generators()
        {x, z}

        """
    group = self.group
    gens = set()
    for syllable in self.array_form:
        gens.add(group.dtype(((syllable[0], 1),)))
    return set(gens)