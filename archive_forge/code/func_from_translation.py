from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@classmethod
def from_translation(cls: type[Mat4T], vector: Vec3) -> Mat4T:
    """Create a translation matrix from a Vec3."""
    return cls((1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, vector[0], vector[1], vector[2], 1.0))