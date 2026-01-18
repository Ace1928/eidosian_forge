from typing import (
from numbers import Integral, Real
@cython.cfunc
@cython.locals(j=cython.int, n=cython.int, x1=cython.double, x2=cython.double, d1=cython.double, d2=cython.double, scale=cython.double, x=cython.double, d=cython.double)
def iup_segment(coords: _PointSegment, rc1: _Point, rd1: _Delta, rc2: _Point, rd2: _Delta):
    """Given two reference coordinates `rc1` & `rc2` and their respective
    delta vectors `rd1` & `rd2`, returns interpolated deltas for the set of
    coordinates `coords`."""
    out_arrays = [None, None]
    for j in (0, 1):
        out_arrays[j] = out = []
        x1, x2, d1, d2 = (rc1[j], rc2[j], rd1[j], rd2[j])
        if x1 == x2:
            n = len(coords)
            if d1 == d2:
                out.extend([d1] * n)
            else:
                out.extend([0] * n)
            continue
        if x1 > x2:
            x1, x2 = (x2, x1)
            d1, d2 = (d2, d1)
        scale = (d2 - d1) / (x2 - x1)
        for pair in coords:
            x = pair[j]
            if x <= x1:
                d = d1
            elif x >= x2:
                d = d2
            else:
                d = d1 + (x - x1) * scale
            out.append(d)
    return zip(*out_arrays)