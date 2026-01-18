from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_equal(self, other):
    return self.apply('eq', self, other, mode='I')