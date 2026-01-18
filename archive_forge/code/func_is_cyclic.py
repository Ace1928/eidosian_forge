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
def is_cyclic(self):
    """
        Return ``True`` if the group is Cyclic.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AbelianGroup
        >>> G = AbelianGroup(3, 4)
        >>> G.is_cyclic
        True
        >>> G = AbelianGroup(4, 4)
        >>> G.is_cyclic
        False

        Notes
        =====

        If the order of a group $n$ can be factored into the distinct
        primes $p_1, p_2, \\dots , p_s$ and if

        .. math::
            \\forall i, j \\in \\{1, 2, \\dots, s \\}:
            p_i \\not \\equiv 1 \\pmod {p_j}

        holds true, there is only one group of the order $n$ which
        is a cyclic group [1]_. This is a generalization of the lemma
        that the group of order $15, 35, \\dots$ are cyclic.

        And also, these additional lemmas can be used to test if a
        group is cyclic if the order of the group is already found.

        - If the group is abelian and the order of the group is
          square-free, the group is cyclic.
        - If the order of the group is less than $6$ and is not $4$, the
          group is cyclic.
        - If the order of the group is prime, the group is cyclic.

        References
        ==========

        .. [1] 1978: John S. Rose: A Course on Group Theory,
            Introduction to Finite Group Theory: 1.4
        """
    if self._is_cyclic is not None:
        return self._is_cyclic
    if len(self.generators) == 1:
        self._is_cyclic = True
        self._is_abelian = True
        return True
    if self._is_abelian is False:
        self._is_cyclic = False
        return False
    order = self.order()
    if order < 6:
        self._is_abelian = True
        if order != 4:
            self._is_cyclic = True
            return True
    factors = factorint(order)
    if all((v == 1 for v in factors.values())):
        if self._is_abelian:
            self._is_cyclic = True
            return True
        primes = list(factors.keys())
        if PermutationGroup._distinct_primes_lemma(primes) is True:
            self._is_cyclic = True
            self._is_abelian = True
            return True
    if not self.is_abelian:
        self._is_cyclic = False
        return False
    self._is_cyclic = all((any((g ** (order // p) != self.identity for g in self.generators)) for p, e in factors.items() if e > 1))
    return self._is_cyclic