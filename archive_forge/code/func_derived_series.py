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
def derived_series(self):
    """Return the derived series for the group.

        Explanation
        ===========

        The derived series for a group `G` is defined as
        `G = G_0 > G_1 > G_2 > \\ldots` where `G_i = [G_{i-1}, G_{i-1}]`,
        i.e. `G_i` is the derived subgroup of `G_{i-1}`, for
        `i\\in\\mathbb{N}`. When we have `G_k = G_{k-1}` for some
        `k\\in\\mathbb{N}`, the series terminates.

        Returns
        =======

        A list of permutation groups containing the members of the derived
        series in the order `G = G_0, G_1, G_2, \\ldots`.

        Examples
        ========

        >>> from sympy.combinatorics.named_groups import (SymmetricGroup,
        ... AlternatingGroup, DihedralGroup)
        >>> A = AlternatingGroup(5)
        >>> len(A.derived_series())
        1
        >>> S = SymmetricGroup(4)
        >>> len(S.derived_series())
        4
        >>> S.derived_series()[1].is_subgroup(AlternatingGroup(4))
        True
        >>> S.derived_series()[2].is_subgroup(DihedralGroup(2))
        True

        See Also
        ========

        derived_subgroup

        """
    res = [self]
    current = self
    nxt = self.derived_subgroup()
    while not current.is_subgroup(nxt):
        res.append(nxt)
        current = nxt
        nxt = nxt.derived_subgroup()
    return res