from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@property
def rank_binary(self):
    """
        Computes the binary ordered rank.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([], ['a','b','c','d'])
        >>> a.rank_binary
        0
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_binary
        3

        See Also
        ========

        iterate_binary, unrank_binary
        """
    if self._rank_binary is None:
        self._rank_binary = int(''.join(Subset.bitlist_from_subset(self.subset, self.superset)), 2)
    return self._rank_binary