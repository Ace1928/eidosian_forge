from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_berk(M, K):
    m = len(M)
    if not m:
        return [[K.one]]
    n = len(M[0])
    if m != n:
        raise DMShapeError('Not square')
    if n == 1:
        return [[K.one], [-M[0][0]]]
    a = M[0][0]
    R = [M[0][1:]]
    C = [[row[0]] for row in M[1:]]
    A = [row[1:] for row in M[1:]]
    q = ddm_berk(A, K)
    T = [[K.zero] * n for _ in range(n + 1)]
    for i in range(n):
        T[i][i] = K.one
        T[i + 1][i] = -a
    for i in range(2, n + 1):
        if i == 2:
            AnC = C
        else:
            C = AnC
            AnC = [[K.zero] for row in C]
            ddm_imatmul(AnC, A, C)
        RAnC = [[K.zero]]
        ddm_imatmul(RAnC, R, AnC)
        for j in range(0, n + 1 - i):
            T[i + j][j] = -RAnC[0][0]
    qout = [[K.zero] for _ in range(n + 1)]
    ddm_imatmul(qout, T, q)
    return qout