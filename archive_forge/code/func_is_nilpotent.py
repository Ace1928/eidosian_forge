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
def is_nilpotent(self):
    """Test if the group is nilpotent.

        Explanation
        ===========

        A group `G` is nilpotent if it has a central series of finite length.
        Alternatively, `G` is nilpotent if its lower central series terminates
        with the trivial group. Every nilpotent group is also solvable
        ([1], p.29, [12]).

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... CyclicGroup)
        >>> C = CyclicGroup(6)
        >>> C.is_nilpotent
        True
        >>> S = SymmetricGroup(5)
        >>> S.is_nilpotent
        False

        See Also
        ========

        lower_central_series, is_solvable

        """
    if self._is_nilpotent is None:
        lcs = self.lower_central_series()
        terminator = lcs[len(lcs) - 1]
        gens = terminator.generators
        degree = self.degree
        identity = _af_new(list(range(degree)))
        if all((g == identity for g in gens)):
            self._is_solvable = True
            self._is_nilpotent = True
            return True
        else:
            self._is_nilpotent = False
            return False
    else:
        return self._is_nilpotent