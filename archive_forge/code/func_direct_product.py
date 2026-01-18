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
@classmethod
def direct_product(cls, *args):
    """
        Returns a TensorSymmetry object that is being a direct product of
        fully (anti-)symmetric index permutation groups.

        Notes
        =====

        Some examples for different values of ``(*args)``:
        ``(1)``         vector, equivalent to ``TensorSymmetry.fully_symmetric(1)``
        ``(2)``         tensor with 2 symmetric indices, equivalent to ``.fully_symmetric(2)``
        ``(-2)``        tensor with 2 antisymmetric indices, equivalent to ``.fully_symmetric(-2)``
        ``(2, -2)``     tensor with the first 2 indices commuting and the last 2 anticommuting
        ``(1, 1, 1)``   tensor with 3 indices without any symmetry
        """
    base, sgs = ([], [Permutation(1)])
    for arg in args:
        if arg > 0:
            bsgs2 = get_symmetric_group_sgs(arg, False)
        elif arg < 0:
            bsgs2 = get_symmetric_group_sgs(-arg, True)
        else:
            continue
        base, sgs = bsgs_direct_product(base, sgs, *bsgs2)
    return TensorSymmetry(base, sgs)