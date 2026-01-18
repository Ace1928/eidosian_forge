from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def reduce_order(func):
    """
    Given the hypergeometric function ``func``, find a sequence of operators to
    reduces order as much as possible.

    Explanation
    ===========

    Return (newfunc, [operators]), where applying the operators to the
    hypergeometric function newfunc yields func.

    Examples
    ========

    >>> from sympy.simplify.hyperexpand import reduce_order, Hyper_Function
    >>> reduce_order(Hyper_Function((1, 2), (3, 4)))
    (Hyper_Function((1, 2), (3, 4)), [])
    >>> reduce_order(Hyper_Function((1,), (1,)))
    (Hyper_Function((), ()), [<Reduce order by cancelling upper 1 with lower 1.>])
    >>> reduce_order(Hyper_Function((2, 4), (3, 3)))
    (Hyper_Function((2,), (3,)), [<Reduce order by cancelling
    upper 4 with lower 3.>])
    """
    nap, nbq, operators = _reduce_order(func.ap, func.bq, ReduceOrder, default_sort_key)
    return (Hyper_Function(Tuple(*nap), Tuple(*nbq)), operators)