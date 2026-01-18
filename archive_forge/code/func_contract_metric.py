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
def contract_metric(self, g):
    """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g : metric

        Notes
        =====

        See the ``TensorIndexType`` docstring for the contraction conventions.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m0)*q(m1)*g(-m0, -m1)
        >>> t.canon_bp()
        metric(L_0, L_1)*p(-L_0)*q(-L_1)
        >>> t.contract_metric(g).canon_bp()
        p(L_0)*q(-L_0)
        """
    expr = self.expand()
    if self != expr:
        expr = canon_bp(expr)
        return contract_metric(expr, g)
    pos_map = self._get_indices_to_args_pos()
    args = list(self.args)
    if g.symmetry == TensorSymmetry.fully_symmetric(-2):
        antisym = 1
    elif g.symmetry == TensorSymmetry.fully_symmetric(2):
        antisym = 0
    elif g.symmetry == TensorSymmetry.no_symmetry(2):
        antisym = None
    else:
        raise NotImplementedError
    gpos = [i for i, x in enumerate(self.args) if isinstance(x, Tensor) and x.component == g]
    if not gpos:
        return self
    sign = 1
    dum = self.dum[:]
    free = self.free[:]
    elim = set()
    for gposx in gpos:
        if gposx in elim:
            continue
        free1 = [x for x in free if pos_map[x[1]] == gposx]
        dum1 = [x for x in dum if pos_map[x[0]] == gposx or pos_map[x[1]] == gposx]
        if not dum1:
            continue
        elim.add(gposx)
        args[gposx] = 1
        if len(dum1) == 2:
            if not antisym:
                dum10, dum11 = dum1
                if pos_map[dum10[1]] == gposx:
                    p0 = dum10[0]
                else:
                    p0 = dum10[1]
                if pos_map[dum11[1]] == gposx:
                    p1 = dum11[0]
                else:
                    p1 = dum11[1]
                dum.append((p0, p1))
            else:
                dum10, dum11 = dum1
                if pos_map[dum10[1]] == gposx:
                    p0 = dum10[0]
                    if dum10[1] == 1:
                        sign = -sign
                else:
                    p0 = dum10[1]
                    if dum10[0] == 0:
                        sign = -sign
                if pos_map[dum11[1]] == gposx:
                    p1 = dum11[0]
                    sign = -sign
                else:
                    p1 = dum11[1]
                dum.append((p0, p1))
        elif len(dum1) == 1:
            if not antisym:
                dp0, dp1 = dum1[0]
                if pos_map[dp0] == pos_map[dp1]:
                    typ = g.index_types[0]
                    sign = sign * typ.dim
                else:
                    if pos_map[dp0] == gposx:
                        p1 = dp1
                    else:
                        p1 = dp0
                    ind, p = free1[0]
                    free.append((ind, p1))
            else:
                dp0, dp1 = dum1[0]
                if pos_map[dp0] == pos_map[dp1]:
                    typ = g.index_types[0]
                    sign = sign * typ.dim
                    if dp0 < dp1:
                        sign = -sign
                else:
                    if pos_map[dp0] == gposx:
                        p1 = dp1
                        if dp0 == 0:
                            sign = -sign
                    else:
                        p1 = dp0
                    ind, p = free1[0]
                    free.append((ind, p1))
        dum = [x for x in dum if x not in dum1]
        free = [x for x in free if x not in free1]
    shift = 0
    shifts = [0] * len(args)
    for i in range(len(args)):
        if i in elim:
            shift += 2
            continue
        shifts[i] = shift
    free = [(ind, p - shifts[pos_map[p]]) for ind, p in free if pos_map[p] not in elim]
    dum = [(p0 - shifts[pos_map[p0]], p1 - shifts[pos_map[p1]]) for i, (p0, p1) in enumerate(dum) if pos_map[p0] not in elim and pos_map[p1] not in elim]
    res = sign * TensMul(*args).doit()
    if not isinstance(res, TensExpr):
        return res
    im = _IndexStructure.from_components_free_dum(res.components, free, dum)
    return res._set_new_index_structure(im)