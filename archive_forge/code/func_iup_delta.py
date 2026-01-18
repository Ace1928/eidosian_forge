from typing import (
from numbers import Integral, Real
def iup_delta(deltas: _DeltaOrNoneSegment, coords: _PointSegment, ends: _Endpoints) -> _DeltaSegment:
    """For the outline given in `coords`, with contour endpoints given
    in sorted increasing order in `ends`, interpolate any missing
    delta values in delta vector `deltas`.

    Returns fully filled-out delta vector."""
    assert sorted(ends) == ends and len(coords) == (ends[-1] + 1 if ends else 0) + 4
    n = len(coords)
    ends = ends + [n - 4, n - 3, n - 2, n - 1]
    out = []
    start = 0
    for end in ends:
        end += 1
        contour = iup_contour(deltas[start:end], coords[start:end])
        out.extend(contour)
        start = end
    return out