from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod
from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift
@deprecated('\n    The tensorhead() function is deprecated. Use tensor_heads() instead.\n    ', deprecated_since_version='1.5', active_deprecations_target='deprecated-tensorhead')
def tensorhead(name, typ, sym=None, comm=0):
    """
    Function generating tensorhead(s). This method is deprecated,
    use TensorHead constructor or tensor_heads() instead.

    Parameters
    ==========

    name : name or sequence of names (as in ``symbols``)

    typ :  index types

    sym :  same as ``*args`` in ``tensorsymmetry``

    comm : commutation group number
    see ``_TensorManager.set_comm``
    """
    if sym is None:
        sym = [[1] for i in range(len(typ))]
    with ignore_warnings(SymPyDeprecationWarning):
        sym = tensorsymmetry(*sym)
    return TensorHead(name, typ, sym, comm)