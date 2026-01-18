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
def next_trotterjohnson(self):
    """
        Returns the next permutation in Trotter-Johnson order.
        If self is the last permutation it returns None.
        See [4] section 2.4. If it is desired to generate all such
        permutations, they can be generated in order more quickly
        with the ``generate_bell`` function.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> p = Permutation([3, 0, 2, 1])
        >>> p.rank_trotterjohnson()
        4
        >>> p = p.next_trotterjohnson(); p
        Permutation([0, 3, 2, 1])
        >>> p.rank_trotterjohnson()
        5

        See Also
        ========

        rank_trotterjohnson, unrank_trotterjohnson, sympy.utilities.iterables.generate_bell
        """
    pi = self.array_form[:]
    n = len(pi)
    st = 0
    rho = pi[:]
    done = False
    m = n - 1
    while m > 0 and (not done):
        d = rho.index(m)
        for i in range(d, m):
            rho[i] = rho[i + 1]
        par = _af_parity(rho[:m])
        if par == 1:
            if d == m:
                m -= 1
            else:
                pi[st + d], pi[st + d + 1] = (pi[st + d + 1], pi[st + d])
                done = True
        elif d == 0:
            m -= 1
            st += 1
        else:
            pi[st + d], pi[st + d - 1] = (pi[st + d - 1], pi[st + d])
            done = True
    if m == 0:
        return None
    return self._af_new(pi)