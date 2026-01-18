from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def lch_to_hpluv(_hx_tuple: Triplet) -> Triplet:
    l = float(_hx_tuple[0])
    c = float(_hx_tuple[1])
    h = float(_hx_tuple[2])
    if l > 100 - 1e-07:
        return (h, 0, 100)
    if l < 1e-08:
        return (h, 0, 0)
    _hx_max = _max_safe_chroma_for_l(l)
    s = c / _hx_max * 100
    return (h, s, l)