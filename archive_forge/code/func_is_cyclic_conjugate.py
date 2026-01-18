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
def is_cyclic_conjugate(self, w):
    """
        Checks whether words ``self``, ``w`` are cyclic conjugates.

        Examples
        ========

        >>> from sympy.combinatorics import free_group
        >>> F, x, y = free_group("x, y")
        >>> w1 = x**2*y**5
        >>> w2 = x*y**5*x
        >>> w1.is_cyclic_conjugate(w2)
        True
        >>> w3 = x**-1*y**5*x**-1
        >>> w3.is_cyclic_conjugate(w2)
        False

        """
    l1 = len(self)
    l2 = len(w)
    if l1 != l2:
        return False
    w1 = self.identity_cyclic_reduction()
    w2 = w.identity_cyclic_reduction()
    letter1 = w1.letter_form
    letter2 = w2.letter_form
    str1 = ' '.join(map(str, letter1))
    str2 = ' '.join(map(str, letter2))
    if len(str1) != len(str2):
        return False
    return str1 in str2 + ' ' + str2