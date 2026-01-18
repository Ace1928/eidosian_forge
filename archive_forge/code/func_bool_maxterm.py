from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def bool_maxterm(k, variables):
    """
    Return the k-th maxterm.

    Each maxterm is assigned an index based on the opposite
    conventional binary encoding used for minterms. The maxterm
    convention assigns the value 0 to the direct form and 1 to
    the complemented form.

    Parameters
    ==========

    k : int or list of 1's and 0's (complementation pattern)
    variables : list of variables

    Examples
    ========
    >>> from sympy.logic.boolalg import bool_maxterm
    >>> from sympy.abc import x, y, z
    >>> bool_maxterm([1, 0, 1], [x, y, z])
    y | ~x | ~z
    >>> bool_maxterm(6, [x, y, z])
    z | ~x | ~y

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Canonical_normal_form#Indexing_maxterms

    """
    if isinstance(k, int):
        k = ibin(k, len(variables))
    variables = tuple(map(sympify, variables))
    return _convert_to_varsPOS(k, variables)