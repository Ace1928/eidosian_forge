from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_convert(self, mode):
    return _Operand(self.im.convert(mode))