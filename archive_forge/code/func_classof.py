from collections import defaultdict
from collections.abc import Iterable
from inspect import isfunction
from functools import reduce
from sympy.assumptions.refine import refine
from sympy.core import SympifyError, Add
from sympy.core.basic import Atom
from sympy.core.decorators import call_highest_priority
from sympy.core.kind import Kind, NumberKind
from sympy.core.logic import fuzzy_and, FuzzyBool
from sympy.core.mod import Mod
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import Abs, re, im
from .utilities import _dotprodsimp, _simplify
from sympy.polys.polytools import Poly
from sympy.utilities.iterables import flatten, is_sequence
from sympy.utilities.misc import as_int, filldedent
from sympy.tensor.array import NDimArray
from .utilities import _get_intermediate_simp_bool
def classof(A, B):
    """
    Get the type of the result when combining matrices of different types.

    Currently the strategy is that immutability is contagious.

    Examples
    ========

    >>> from sympy import Matrix, ImmutableMatrix
    >>> from sympy.matrices.common import classof
    >>> M = Matrix([[1, 2], [3, 4]]) # a Mutable Matrix
    >>> IM = ImmutableMatrix([[1, 2], [3, 4]])
    >>> classof(M, IM)
    <class 'sympy.matrices.immutable.ImmutableDenseMatrix'>
    """
    priority_A = getattr(A, '_class_priority', None)
    priority_B = getattr(B, '_class_priority', None)
    if None not in (priority_A, priority_B):
        if A._class_priority > B._class_priority:
            return A.__class__
        else:
            return B.__class__
    try:
        import numpy
    except ImportError:
        pass
    else:
        if isinstance(A, numpy.ndarray):
            return B.__class__
        if isinstance(B, numpy.ndarray):
            return A.__class__
    raise TypeError('Incompatible classes %s, %s' % (A.__class__, B.__class__))