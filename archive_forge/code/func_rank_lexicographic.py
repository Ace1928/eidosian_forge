from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
@property
def rank_lexicographic(self):
    """
        Computes the lexicographic ranking of the subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.rank_lexicographic
        14
        >>> a = Subset([2, 4, 5], [1, 2, 3, 4, 5, 6])
        >>> a.rank_lexicographic
        43
        """
    if self._rank_lex is None:

        def _ranklex(self, subset_index, i, n):
            if subset_index == [] or i > n:
                return 0
            if i in subset_index:
                subset_index.remove(i)
                return 1 + _ranklex(self, subset_index, i + 1, n)
            return 2 ** (n - i - 1) + _ranklex(self, subset_index, i + 1, n)
        indices = Subset.subset_indices(self.subset, self.superset)
        self._rank_lex = _ranklex(self, indices, 0, self.superset_size)
    return self._rank_lex