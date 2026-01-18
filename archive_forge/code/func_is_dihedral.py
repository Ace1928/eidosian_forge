from math import factorial as _factorial, log, prod
from itertools import chain, islice, product
from sympy.combinatorics import Permutation
from sympy.combinatorics.permutations import (_af_commutes_with, _af_invert,
from sympy.combinatorics.util import (_check_cycles_alt_sym,
from sympy.core import Basic
from sympy.core.random import _randrange, randrange, choice
from sympy.core.symbol import Symbol
from sympy.core.sympify import _sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.ntheory import primefactors, sieve
from sympy.ntheory.factor_ import (factorint, multiplicity)
from sympy.ntheory.primetest import isprime
from sympy.utilities.iterables import has_variety, is_sequence, uniq
@property
def is_dihedral(self):
    """
        Return ``True`` if the group is dihedral.

        Examples
        ========

        >>> from sympy.combinatorics.perm_groups import PermutationGroup
        >>> from sympy.combinatorics.permutations import Permutation
        >>> from sympy.combinatorics.named_groups import SymmetricGroup, CyclicGroup
        >>> G = PermutationGroup(Permutation(1, 6)(2, 5)(3, 4), Permutation(0, 1, 2, 3, 4, 5, 6))
        >>> G.is_dihedral
        True
        >>> G = SymmetricGroup(3)
        >>> G.is_dihedral
        True
        >>> G = CyclicGroup(6)
        >>> G.is_dihedral
        False

        References
        ==========

        .. [Di1] https://math.stackexchange.com/a/827273
        .. [Di2] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral.pdf
        .. [Di3] https://kconrad.math.uconn.edu/blurbs/grouptheory/dihedral2.pdf
        .. [Di4] https://en.wikipedia.org/wiki/Dihedral_group
        """
    if self._is_dihedral is not None:
        return self._is_dihedral
    order = self.order()
    if order % 2 == 1:
        self._is_dihedral = False
        return False
    if order == 2:
        self._is_dihedral = True
        return True
    if order == 4:
        self._is_dihedral = not self.is_cyclic
        return self._is_dihedral
    if self.is_abelian:
        self._is_dihedral = False
        return False
    n = order // 2
    gens = self.generators
    if len(gens) == 2:
        x, y = gens
        a, b = (x.order(), y.order())
        if a < b:
            x, y, a, b = (y, x, b, a)
        if a == 2 == b:
            self._is_dihedral = True
            return True
        if a == n and b == 2 and (y * x * y == ~x):
            self._is_dihedral = True
            return True
    order_2, order_n = ([], [])
    for p in self.elements:
        k = p.order()
        if k == 2:
            order_2.append(p)
        elif k == n:
            order_n.append(p)
    if len(order_2) != n + 1 - n % 2:
        self._is_dihedral = False
        return False
    if not order_n:
        self._is_dihedral = False
        return False
    x = order_n[0]
    y = order_2[0]
    if n % 2 == 0 and y == x ** (n // 2):
        y = order_2[1]
    self._is_dihedral = y * x * y == ~x
    return self._is_dihedral