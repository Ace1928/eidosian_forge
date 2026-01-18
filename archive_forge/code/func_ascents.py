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
def ascents(self):
    """
        Returns the positions of ascents in a permutation, ie, the location
        where p[i] < p[i+1]

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> p = Permutation([4, 0, 1, 3, 2])
        >>> p.ascents()
        [1, 2]

        See Also
        ========

        descents, inversions, min, max
        """
    a = self.array_form
    pos = [i for i in range(len(a) - 1) if a[i] < a[i + 1]]
    return pos