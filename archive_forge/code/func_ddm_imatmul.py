from __future__ import annotations
from operator import mul
from .exceptions import (
from typing import Sequence, TypeVar
from sympy.polys.matrices._typing import RingElement
def ddm_imatmul(a: list[list[R]], b: Sequence[Sequence[R]], c: Sequence[Sequence[R]]) -> None:
    """a += b @ c"""
    cT = list(zip(*c))
    for bi, ai in zip(b, a):
        for j, cTj in enumerate(cT):
            ai[j] = sum(map(mul, bi, cTj), ai[j])