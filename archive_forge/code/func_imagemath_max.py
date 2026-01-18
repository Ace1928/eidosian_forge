from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_max(self, other):
    return self.apply('max', self, other)