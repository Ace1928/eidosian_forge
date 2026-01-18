from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_idet(a, K):
    """a  <--  echelon(a); return det"""
    m = len(a)
    if not m:
        return K.one
    n = len(a[0])
    exquo = K.exquo
    uf = K.one
    for k in range(n - 1):
        if not a[k][k]:
            for i in range(k + 1, n):
                if a[i][k]:
                    a[k], a[i] = (a[i], a[k])
                    uf = -uf
                    break
            else:
                return K.zero
        akkm1 = a[k - 1][k - 1] if k else K.one
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                a[i][j] = exquo(a[i][j] * a[k][k] - a[i][k] * a[k][j], akkm1)
    return uf * a[-1][-1]