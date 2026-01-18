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
def power_of(self, other):
    """
        Check if `self == other**n` for some integer n.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> ((x*y)**2).power_of(x*y)
        True
        >>> (x**-3*y**-2*x**3).power_of(x**-3*y*x**3)
        True

        """
    if self.is_identity:
        return True
    l = len(other)
    if l == 1:
        gens = self.contains_generators()
        s = other in gens or other ** (-1) in gens
        return len(gens) == 1 and s
    reduced, r1 = self.cyclic_reduction(removed=True)
    if not r1.is_identity:
        other, r2 = other.cyclic_reduction(removed=True)
        if r1 == r2:
            return reduced.power_of(other)
        return False
    if len(self) < l or len(self) % l:
        return False
    prefix = self.subword(0, l)
    if prefix == other or prefix ** (-1) == other:
        rest = self.subword(l, len(self))
        return rest.power_of(other)
    return False