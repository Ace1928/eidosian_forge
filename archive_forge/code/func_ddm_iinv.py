from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_iinv(ainv, a, K):
    if not K.is_Field:
        raise ValueError('Not a field')
    m = len(a)
    if not m:
        return
    n = len(a[0])
    if m != n:
        raise DMNonSquareMatrixError
    eye = [[K.one if i == j else K.zero for j in range(n)] for i in range(n)]
    Aaug = [row + eyerow for row, eyerow in zip(a, eye)]
    pivots = ddm_irref(Aaug)
    if pivots != list(range(n)):
        raise DMNonInvertibleMatrixError('Matrix det == 0; not invertible.')
    ainv[:] = [row[n:] for row in Aaug]