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
def is_transitive(self, strict=True):
    """Test if the group is transitive.

        Explanation
        ===========

        A group is transitive if it has a single orbit.

        If ``strict`` is ``False`` the group is transitive if it has
        a single orbit of length different from 1.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([2, 0, 1, 3])
        >>> G1 = PermutationGroup([a, b])
        >>> G1.is_transitive()
        False
        >>> G1.is_transitive(strict=False)
        True
        >>> c = Permutation([2, 3, 0, 1])
        >>> G2 = PermutationGroup([a, c])
        >>> G2.is_transitive()
        True
        >>> d = Permutation([1, 0, 2, 3])
        >>> e = Permutation([0, 1, 3, 2])
        >>> G3 = PermutationGroup([d, e])
        >>> G3.is_transitive() or G3.is_transitive(strict=False)
        False

        """
    if self._is_transitive:
        return self._is_transitive
    if strict:
        if self._is_transitive is not None:
            return self._is_transitive
        ans = len(self.orbit(0)) == self.degree
        self._is_transitive = ans
        return ans
    got_orb = False
    for x in self.orbits():
        if len(x) > 1:
            if got_orb:
                return False
            got_orb = True
    return got_orb