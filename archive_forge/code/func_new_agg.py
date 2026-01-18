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
def new_agg():
    return np.zeros((5, 5), dtype='i4')