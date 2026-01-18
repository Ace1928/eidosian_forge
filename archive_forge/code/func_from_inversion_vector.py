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
@classmethod
def from_inversion_vector(cls, inversion):
    """
        Calculates the permutation from the inversion vector.

        Examples
        ========

        >>> from sympy.combinatorics import Permutation
        >>> from sympy import init_printing
        >>> init_printing(perm_cyclic=False, pretty_print=False)
        >>> Permutation.from_inversion_vector([3, 2, 1, 0, 0])
        Permutation([3, 2, 1, 0, 4, 5])

        """
    size = len(inversion)
    N = list(range(size + 1))
    perm = []
    try:
        for k in range(size):
            val = N[inversion[k]]
            perm.append(val)
            N.remove(val)
    except IndexError:
        raise ValueError('The inversion vector is not valid.')
    perm.extend(N)
    return cls._af_new(perm)