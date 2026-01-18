import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
def rank_nonlex(self, inv_perm=None):
    """
        This is a linear time ranking algorithm that does not
        enforce lexicographic order [3].


        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([0, 1, 2, 3])
        >>> p.rank_nonlex()
        23

        See Also
        ========

        next_nonlex, unrank_nonlex
        """

    def _rank1(n, perm, inv_perm):
        if n == 1:
            return 0
        s = perm[n - 1]
        t = inv_perm[n - 1]
        perm[n - 1], perm[t] = (perm[t], s)
        inv_perm[n - 1], inv_perm[s] = (inv_perm[s], t)
        return s + n * _rank1(n - 1, perm, inv_perm)
    if inv_perm is None:
        inv_perm = (~self).array_form
    if not inv_perm:
        return 0
    perm = self.array_form[:]
    r = _rank1(len(perm), perm, inv_perm)
    return r