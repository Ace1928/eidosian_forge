from typing import (
from numbers import Integral, Real
def iup_delta_optimize(deltas: _DeltaSegment, coords: _PointSegment, ends: _Endpoints, tolerance: Real=0.0) -> _DeltaOrNoneSegment:
    """For the outline given in `coords`, with contour endpoints given
    in sorted increasing order in `ends`, optimize a set of delta
    values `deltas` within error `tolerance`.

    Returns delta vector that has most number of None items instead of
    the input delta.
    """
    assert sorted(ends) == ends and len(coords) == (ends[-1] + 1 if ends else 0) + 4
    n = len(coords)
    ends = ends + [n - 4, n - 3, n - 2, n - 1]
    out = []
    start = 0
    for end in ends:
        contour = iup_contour_optimize(deltas[start:end + 1], coords[start:end + 1], tolerance)
        assert len(contour) == end - start + 1
        out.extend(contour)
        start = end + 1
    return out