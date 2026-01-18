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
def is_elementary(self, p):
    """Return ``True`` if the group is elementary abelian. An elementary
        abelian group is a finite abelian group, where every nontrivial
        element has order `p`, where `p` is a prime.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> a = Permutation([0, 2, 1])
        >>> G = PermutationGroup([a])
        >>> G.is_elementary(2)
        True
        >>> a = Permutation([0, 2, 1, 3])
        >>> b = Permutation([3, 1, 2, 0])
        >>> G = PermutationGroup([a, b])
        >>> G.is_elementary(2)
        True
        >>> G.is_elementary(3)
        False

        """
    return self.is_abelian and all((g.order() == p for g in self.generators))