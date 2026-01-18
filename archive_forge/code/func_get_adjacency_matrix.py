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
def get_adjacency_matrix(self):
    """
        Computes the adjacency matrix of a permutation.

        Explanation
        ===========

        If job i is adjacent to job j in a permutation p
        then we set m[i, j] = 1 where m is the adjacency
        matrix of p.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation.josephus(3, 6, 1)
        >>> p.get_adjacency_matrix()
        Matrix([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0]])
        >>> q = Permutation([0, 1, 2, 3])
        >>> q.get_adjacency_matrix()
        Matrix([
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 0, 0, 0]])

        See Also
        ========

        get_precedence_matrix, get_precedence_distance, get_adjacency_distance
        """
    m = zeros(self.size)
    perm = self.array_form
    for i in range(self.size - 1):
        m[perm[i], perm[i + 1]] = 1
    return m