from itertools import combinations
from sympy.combinatorics.graycode import GrayCode
def next_gray(self):
    """
        Generates the next Gray code ordered subset.

        Examples
        ========

        >>> from sympy.combinatorics import Subset
        >>> a = Subset([1, 2, 3], [1, 2, 3, 4])
        >>> a.next_gray().subset
        [1, 3]

        See Also
        ========

        iterate_graycode, prev_gray
        """
    return self.iterate_graycode(1)