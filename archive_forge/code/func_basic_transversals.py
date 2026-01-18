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
def basic_transversals(self):
    """
        Return basic transversals relative to a base and strong generating set.

        Explanation
        ===========

        The basic transversals are transversals of the basic orbits. They
        are provided as a list of dictionaries, each dictionary having
        keys - the elements of one of the basic orbits, and values - the
        corresponding transversal elements. See [1], pp. 87-89 for more
        information.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import AlternatingGroup
        >>> A = AlternatingGroup(4)
        >>> A.basic_transversals
        [{0: (3), 1: (3)(0 1 2), 2: (3)(0 2 1), 3: (0 3 1)}, {1: (3), 2: (1 2 3), 3: (1 3 2)}]

        See Also
        ========

        strong_gens, base, basic_orbits, basic_stabilizers

        """
    if self._transversals == []:
        self.schreier_sims()
    return self._transversals