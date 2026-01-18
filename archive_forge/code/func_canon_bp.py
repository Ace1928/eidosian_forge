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
def canon_bp(self):
    """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        >>> t = A(m0,-m1)*A(m1,-m0)
        >>> t.canon_bp()
        -A(L_0, L_1)*A(-L_0, -L_1)
        >>> t = A(m0,-m1)*A(m1,-m2)*A(m2,-m0)
        >>> t.canon_bp()
        0
        """
    if self._is_canon_bp:
        return self
    expr = self.expand()
    if isinstance(expr, TensAdd):
        return expr.canon_bp()
    if not expr.components:
        return expr
    t = expr.sorted_components()
    g, dummies, msym = t._index_structure.indices_canon_args()
    v = components_canon_args(t.components)
    can = canonicalize(g, dummies, msym, *v)
    if can == 0:
        return S.Zero
    tmul = t.perm2tensor(can, True)
    return tmul