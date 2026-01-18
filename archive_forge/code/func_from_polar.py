from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@staticmethod
def from_polar(mag: float, angle: float) -> Vec2:
    """Create a new vector from the given polar coordinates.

        :parameters:
            `mag`   : int or float :
                The magnitude of the vector.
            `angle` : int or float :
                The angle of the vector in radians.
        """
    return Vec2(mag * _math.cos(angle), mag * _math.sin(angle))