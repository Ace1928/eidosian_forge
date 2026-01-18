from __future__ import annotations
from typing import Any, Literal, overload
import numpy
def delta_decode(data: bytes | bytearray | numpy.ndarray, /, axis: int=-1, dist: int=1, *, out=None) -> bytes | numpy.ndarray:
    """Decode Delta."""
    if dist != 1:
        raise NotImplementedError(f"delta_decode with dist={dist!r} requires the 'imagecodecs' package")
    if out is not None and (not out.flags.writeable):
        out = None
    if isinstance(data, (bytes, bytearray)):
        data = numpy.frombuffer(data, dtype=numpy.uint8)
        return numpy.cumsum(data, axis=0, dtype=numpy.uint8, out=out).tobytes()
    if data.dtype.kind == 'f':
        if not data.dtype.isnative:
            raise NotImplementedError(f"delta_decode with {data.dtype!r} requires the 'imagecodecs' package")
        view = data.view(f'{data.dtype.byteorder}u{data.dtype.itemsize}')
        view = numpy.cumsum(view, axis=axis, dtype=view.dtype)
        return view.view(data.dtype)
    return numpy.cumsum(data, axis=axis, dtype=data.dtype, out=out)