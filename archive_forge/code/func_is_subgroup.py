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
def is_subgroup(self, G, strict=True):
    """Return ``True`` if all elements of ``self`` belong to ``G``.

        If ``strict`` is ``False`` then if ``self``'s degree is smaller
        than ``G``'s, the elements will be resized to have the same degree.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> from sympy.combinatorics import SymmetricGroup, CyclicGroup

        Testing is strict by default: the degree of each group must be the
        same:

        >>> p = Permutation(0, 1, 2, 3, 4, 5)
        >>> G1 = PermutationGroup([Permutation(0, 1, 2), Permutation(0, 1)])
        >>> G2 = PermutationGroup([Permutation(0, 2), Permutation(0, 1, 2)])
        >>> G3 = PermutationGroup([p, p**2])
        >>> assert G1.order() == G2.order() == G3.order() == 6
        >>> G1.is_subgroup(G2)
        True
        >>> G1.is_subgroup(G3)
        False
        >>> G3.is_subgroup(PermutationGroup(G3[1]))
        False
        >>> G3.is_subgroup(PermutationGroup(G3[0]))
        True

        To ignore the size, set ``strict`` to ``False``:

        >>> S3 = SymmetricGroup(3)
        >>> S5 = SymmetricGroup(5)
        >>> S3.is_subgroup(S5, strict=False)
        True
        >>> C7 = CyclicGroup(7)
        >>> G = S5*C7
        >>> S5.is_subgroup(G, False)
        True
        >>> C7.is_subgroup(G, 0)
        False

        """
    if isinstance(G, SymmetricPermutationGroup):
        if self.degree != G.degree:
            return False
        return True
    if not isinstance(G, PermutationGroup):
        return False
    if self == G or self.generators[0] == Permutation():
        return True
    if G.order() % self.order() != 0:
        return False
    if self.degree == G.degree or (self.degree < G.degree and (not strict)):
        gens = self.generators
    else:
        return False
    return all((G.contains(g, strict=strict) for g in gens))