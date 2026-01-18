from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_min(self, other):
    return self.apply('min', self, other)