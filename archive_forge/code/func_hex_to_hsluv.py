from __future__ import annotations
import math as _math  # unexport, see #17
import typing
from functools import partial as _partial
from functools import wraps as _wraps  # unexport, see #17
def hex_to_hsluv(s: RGBHexColor) -> Triplet:
    return rgb_to_hsluv(hex_to_rgb(s))