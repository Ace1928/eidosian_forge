from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_irref(a, _partial_pivot=False):
    """a  <--  rref(a)"""
    m = len(a)
    if not m:
        return []
    n = len(a[0])
    i = 0
    pivots = []
    for j in range(n):
        if _partial_pivot:
            ip = max(range(i, m), key=lambda ip: abs(a[ip][j]))
            a[i], a[ip] = (a[ip], a[i])
        aij = a[i][j]
        if not aij:
            for ip in range(i + 1, m):
                aij = a[ip][j]
                if aij:
                    a[i], a[ip] = (a[ip], a[i])
                    break
            else:
                continue
        ai = a[i]
        aijinv = aij ** (-1)
        for l in range(j, n):
            ai[l] *= aijinv
        for k, ak in enumerate(a):
            if k == i or not ak[j]:
                continue
            akj = ak[j]
            ak[j] -= akj
            for l in range(j + 1, n):
                ak[l] -= akj * ai[l]
        pivots.append(j)
        i += 1
        if i >= m:
            break
    return pivots