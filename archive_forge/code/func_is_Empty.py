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
@property
def is_Empty(self):
    """
        Checks to see if the permutation is a set with zero elements

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> Permutation([]).is_Empty
        True
        >>> Permutation([0]).is_Empty
        False

        See Also
        ========

        is_Singleton
        """
    return self.size == 0