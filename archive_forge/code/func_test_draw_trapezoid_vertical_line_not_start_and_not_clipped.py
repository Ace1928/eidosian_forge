from __future__ import annotations
import pandas as pd
import numpy as np
import pytest
from datashader.datashape import dshape
from datashader.glyphs import Point, LinesAxis1, Glyph
from datashader.glyphs.area import _build_draw_trapezoid_y
from datashader.glyphs.line import (
from datashader.glyphs.trimesh import(
from datashader.utils import ngjit
def test_draw_trapezoid_vertical_line_not_start_and_not_clipped():
    x0, x1 = (2, 2)
    y0, y1, y2, y3 = (1, 3, 4, 0)
    trapezoid_start = False
    stacked = True
    out = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_trapezoid(x0, x1, y0, y1, y2, y3, 0, trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)
    out = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    agg = new_agg()
    draw_trapezoid(x1, x0, y3, y2, y1, y0, 0, trapezoid_start, stacked, agg)
    np.testing.assert_equal(agg, out)