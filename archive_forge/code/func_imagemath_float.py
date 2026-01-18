from __future__ import annotations
import builtins
from . import Image, _imagingmath
def imagemath_float(self):
    return _Operand(self.im.convert('F'))