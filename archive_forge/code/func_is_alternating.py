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
def is_alternating(self):
    """Return ``True`` if the group is alternating.

        Examples
        ========

        >>> from sympy.combinatorics import AlternatingGroup
        >>> g = AlternatingGroup(5)
        >>> g.is_alternating
        True

        >>> from sympy.combinatorics import Permutation, PermutationGroup
        >>> g = PermutationGroup(
        ...     Permutation(0, 1, 2, 3, 4),
        ...     Permutation(2, 3, 4))
        >>> g.is_alternating
        True

        Notes
        =====

        This uses a naive test involving the computation of the full
        group order.
        If you need more quicker taxonomy for large groups, you can use
        :meth:`PermutationGroup.is_alt_sym`.
        However, :meth:`PermutationGroup.is_alt_sym` may not be accurate
        and is not able to distinguish between an alternating group and
        a symmetric group.

        See Also
        ========

        is_alt_sym
        """
    _is_alt = self._is_alt
    if _is_alt is not None:
        return _is_alt
    n = self.degree
    if n >= 8:
        if self.is_transitive():
            _is_alt_sym = self._eval_is_alt_sym_monte_carlo()
            if _is_alt_sym:
                if all((g.is_even for g in self.generators)):
                    self._is_sym, self._is_alt = (False, True)
                    return True
                self._is_sym, self._is_alt = (True, False)
                return False
            return self._eval_is_alt_sym_naive(only_alt=True)
        self._is_sym, self._is_alt = (False, False)
        return False
    return self._eval_is_alt_sym_naive(only_alt=True)