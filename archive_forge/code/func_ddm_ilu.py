from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_ilu(a):
    """a  <--  LU(a)"""
    m = len(a)
    if not m:
        return []
    n = len(a[0])
    swaps = []
    for i in range(min(m, n)):
        if not a[i][i]:
            for ip in range(i + 1, m):
                if a[ip][i]:
                    swaps.append((i, ip))
                    a[i], a[ip] = (a[ip], a[i])
                    break
            else:
                continue
        for j in range(i + 1, m):
            l_ji = a[j][i] / a[i][i]
            a[j][i] = l_ji
            for k in range(i + 1, n):
                a[j][k] -= l_ji * a[i][k]
    return swaps