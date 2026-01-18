from __future__ import annotations
from math import floor
import numpy as np
from toolz import memoize
from datashader.glyphs.points import _PointLike
from datashader.utils import isreal, ngjit
@ngjit
def extend_triangles(vt, bounds, verts, weight_type, interpolate, aggs, cols):
    """Aggregate along an array of triangles formed by arrays of CW
        vertices. Each row corresponds to a single triangle definition.

        `weight_type == True` means "weights are on vertices"
        """
    xmin, xmax, ymin, ymax = bounds
    cmax_x, cmax_y = (max(xmin, xmax), max(ymin, ymax))
    cmin_x, cmin_y = (min(xmin, xmax), min(ymin, ymax))
    vmax_x, vmax_y = map_onto_pixel(vt, bounds, cmax_x, cmax_y)
    vmin_x, vmin_y = map_onto_pixel(vt, bounds, cmin_x, cmin_y)
    max_x_pixels = round((bounds[1] - bounds[0]) * vt[0]) - 1
    max_y_pixels = round((bounds[3] - bounds[2]) * vt[2]) - 1
    col = cols[0]
    n_tris = verts.shape[0]
    for n in range(0, n_tris, 3):
        a = verts[n]
        b = verts[n + 1]
        c = verts[n + 2]
        axn, ayn = (a[0], a[1])
        bxn, byn = (b[0], b[1])
        cxn, cyn = (c[0], c[1])
        col0, col1, col2 = (col[n], col[n + 1], col[n + 2])
        ax, ay = map_onto_pixel(vt, bounds, axn, ayn)
        bx, by = map_onto_pixel(vt, bounds, bxn, byn)
        cx, cy = map_onto_pixel(vt, bounds, cxn, cyn)
        minx = min(ax, bx, cx)
        maxx = max(ax, bx, cx)
        miny = min(ay, by, cy)
        maxy = max(ay, by, cy)
        if minx >= vmax_x or maxx < vmin_x or miny >= vmax_y or (maxy < vmin_y):
            continue
        minx = max(minx, vmin_x)
        maxx = min(maxx, vmax_x)
        miny = max(miny, vmin_y)
        maxy = min(maxy, vmax_y)
        minx = max(floor(minx + 0.5), 0)
        miny = max(floor(miny + 0.5), 0)
        maxx = min(floor(maxx + 0.5), max_x_pixels)
        maxy = min(floor(maxy + 0.5), max_y_pixels)
        bias0, bias1, bias2 = (-1, -1, -1)
        if ay < by or (by == ay and ax < bx):
            bias0 = 0
        if by < cy or (cy == by and bx < cx):
            bias1 = 0
        if cy < ay or (ay == cy and cx < ax):
            bias2 = 0
        bbox = (minx, maxx, miny, maxy)
        biases = (bias0, bias1, bias2)
        mapped_verts = ((ax, ay), (bx, by), (cx, cy))
        if interpolate:
            weights = (col0, col1, col2)
            draw_triangle_interp(mapped_verts, bbox, biases, aggs, weights)
        else:
            val = (col[n] + col[n + 1] + col[n + 2]) / 3
            draw_triangle(mapped_verts, bbox, biases, aggs, val)