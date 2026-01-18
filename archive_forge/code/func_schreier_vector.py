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
def schreier_vector(self, alpha):
    """Computes the schreier vector for ``alpha``.

        Explanation
        ===========

        The Schreier vector efficiently stores information
        about the orbit of ``alpha``. It can later be used to quickly obtain
        elements of the group that send ``alpha`` to a particular element
        in the orbit. Notice that the Schreier vector depends on the order
        in which the group generators are listed. For a definition, see [3].
        Since list indices start from zero, we adopt the convention to use
        "None" instead of 0 to signify that an element does not belong
        to the orbit.
        For the algorithm and its correctness, see [2], pp.78-80.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([2, 4, 6, 3, 1, 5, 0])
        >>> b = Permutation([0, 1, 3, 5, 4, 6, 2])
        >>> G = PermutationGroup([a, b])
        >>> G.schreier_vector(0)
        [-1, None, 0, 1, None, 1, 0]

        See Also
        ========

        orbit

        """
    n = self.degree
    v = [None] * n
    v[alpha] = -1
    orb = [alpha]
    used = [False] * n
    used[alpha] = True
    gens = self.generators
    r = len(gens)
    for b in orb:
        for i in range(r):
            temp = gens[i]._array_form[b]
            if used[temp] is False:
                orb.append(temp)
                used[temp] = True
                v[temp] = i
    return v