from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
def iterate_binary(self, k):
    """
        This is a helper function. It iterates over the
        binary subsets by ``k`` steps. This variable can be
        both positive or negative.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset(['c', 'd'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(-2).subset
        ['d']
        >>> a = Subset(['a', 'b', 'c'], ['a', 'b', 'c', 'd'])
        >>> a.iterate_binary(2).subset
        []

        See Also
        ========

        next_binary, prev_binary
        """
    bin_list = Subset.bitlist_from_subset(self.subset, self.superset)
    n = (int(''.join(bin_list), 2) + k) % 2 ** self.superset_size
    bits = bin(n)[2:].rjust(self.superset_size, '0')
    return Subset.subset_from_bitlist(self.superset, bits)