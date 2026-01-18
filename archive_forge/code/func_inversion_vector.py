import random
from collections import defaultdict
from collections.abc import Iterable
from functools import reduce
from sympy.core.parameters import global_parameters
from sympy.core.basic import Atom
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.sympify import _sympify
from sympy.matrices import zeros
from sympy.polys.polytools import lcm
from sympy.printing.repr import srepr
from sympy.utilities.iterables import (flatten, has_variety, minlex,
from sympy.utilities.misc import as_int
from mpmath.libmp.libintmath import ifac
from sympy.multipledispatch import dispatch
def inversion_vector(self):
    """Return the inversion vector of the permutation.

        The inversion vector consists of elements whose value
        indicates the number of elements in the permutation
        that are lesser than it and lie on its right hand side.

        The inversion vector is the same as the Lehmer encoding of a
        permutation.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 8, 0, 7, 1, 5, 3, 6, 2])
        >>> p.inversion_vector()
        [4, 7, 0, 5, 0, 2, 1, 1]
        >>> p = Permutation([3, 2, 1, 0])
        >>> p.inversion_vector()
        [3, 2, 1]

        The inversion vector increases lexicographically with the rank
        of the permutation, the -ith element cycling through 0..i.

        >>> p = Permutation(2)
        >>> while p:
        ...     print('%s %s %s' % (p, p.inversion_vector(), p.rank()))
        ...     p = p.next_lex()
        (2) [0, 0] 0
        (1 2) [0, 1] 1
        (2)(0 1) [1, 0] 2
        (0 1 2) [1, 1] 3
        (0 2 1) [2, 0] 4
        (0 2) [2, 1] 5

        See Also
        ========

        from_inversion_vector
        """
    self_array_form = self.array_form
    n = len(self_array_form)
    inversion_vector = [0] * (n - 1)
    for i in range(n - 1):
        val = 0
        for j in range(i + 1, n):
            if self_array_form[j] < self_array_form[i]:
                val += 1
        inversion_vector[i] = val
    return inversion_vector