from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def packints_encode(data: numpy.ndarray, /, bitspersample: int, axis: int=-1, *, out=None) -> bytes:
    """Tightly pack integers."""
    raise NotImplementedError("packints_encode requires the 'imagecodecs' package")