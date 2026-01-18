from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize
from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit
@ngjit
def draw_triangle(verts, bbox, biases, aggs, val):
    """Draw a triangle on a grid.

        Plots a triangle with integer coordinates onto a pixel grid,
        clipping to the bounds. The vertices are assumed to have
        already been scaled and transformed.
        """
    minx, maxx, miny, maxy = bbox
    if minx == maxx and miny == maxy:
        append(minx, miny, *aggs + (val,))
    else:
        (ax, ay), (bx, by), (cx, cy) = verts
        bias0, bias1, bias2 = biases
        for j in range(miny, maxy + 1):
            for i in range(minx, maxx + 1):
                g2 = edge_func(ax, ay, bx, by, i, j)
                g0 = edge_func(bx, by, cx, cy, i, j)
                g1 = edge_func(cx, cy, ax, ay, i, j)
                if (g2 > 0 or (bias0 < 0 and g2 == 0)) and (g0 > 0 or (bias1 < 0 and g0 == 0)) and (g1 > 0 or (bias2 < 0 and g1 == 0)):
                    append(i, j, *aggs + (val,))