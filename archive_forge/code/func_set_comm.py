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
def set_comm(self, i, j, c):
    """
        Set the commutation parameter ``c`` for commutation groups ``i, j``.

        Parameters
        ==========

        i, j : symbols representing commutation groups

        c  :  group commutation number

        Notes
        =====

        ``i, j`` can be symbols, strings or numbers,
        apart from ``0, 1`` and ``2`` which are reserved respectively
        for commuting, anticommuting tensors and tensors not commuting
        with any other group apart with the commuting tensors.
        For the remaining cases, use this method to set the commutation rules;
        by default ``c=None``.

        The group commutation number ``c`` is assigned in correspondence
        to the group commutation symbols; it can be

        0        commuting

        1        anticommuting

        None     no commutation property

        Examples
        ========

        ``G`` and ``GH`` do not commute with themselves and commute with
        each other; A is commuting.

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorManager, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz')
        >>> i0,i1,i2,i3,i4 = tensor_indices('i0:5', Lorentz)
        >>> A = TensorHead('A', [Lorentz])
        >>> G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
        >>> GH = TensorHead('GH', [Lorentz], TensorSymmetry.no_symmetry(1), 'GHcomm')
        >>> TensorManager.set_comm('Gcomm', 'GHcomm', 0)
        >>> (GH(i1)*G(i0)).canon_bp()
        G(i0)*GH(i1)
        >>> (G(i1)*G(i0)).canon_bp()
        G(i1)*G(i0)
        >>> (G(i1)*A(i0)).canon_bp()
        A(i0)*G(i1)
        """
    if c not in (0, 1, None):
        raise ValueError('`c` can assume only the values 0, 1 or None')
    if i not in self._comm_symbols2i:
        n = len(self._comm)
        self._comm.append({})
        self._comm[n][0] = 0
        self._comm[0][n] = 0
        self._comm_symbols2i[i] = n
        self._comm_i2symbol[n] = i
    if j not in self._comm_symbols2i:
        n = len(self._comm)
        self._comm.append({})
        self._comm[0][n] = 0
        self._comm[n][0] = 0
        self._comm_symbols2i[j] = n
        self._comm_i2symbol[n] = j
    ni = self._comm_symbols2i[i]
    nj = self._comm_symbols2i[j]
    self._comm[ni][nj] = c
    self._comm[nj][ni] = c