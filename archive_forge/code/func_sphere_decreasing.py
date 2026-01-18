from __future__ import annotations
from math import log, pi, sin, sqrt
from ._binary import o8
def sphere_decreasing(middle, pos):
    return 1.0 - sqrt(1.0 - linear(middle, pos) ** 2)