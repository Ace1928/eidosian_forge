from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.property_mixins import FillProps, HatchProps, LineProps
from ..models.glyphs import MultiLine, MultiPolygons
from ..models.renderers import ContourRenderer, GlyphRenderer
from ..models.sources import ColumnDataSource
from ..palettes import interp_palette
from ..plotting._renderer import _process_sequence_literals
from ..util.dataclasses import dataclass, entries
def contour_data(x: ArrayLike | None=None, y: ArrayLike | None=None, z: ArrayLike | np.ma.MaskedArray | None=None, levels: ArrayLike | None=None, *, want_fill: bool=True, want_line: bool=True) -> ContourData:
    """ Return the contour data of filled and/or line contours that can be
    passed to :func:`bokeh.models.ContourRenderer.set_data`
    """
    levels = _validate_levels(levels)
    if len(levels) < 2:
        want_fill = False
    if not want_fill and (not want_line):
        raise ValueError('Neither fill nor line requested in contour_data')
    coords = _contour_coords(x, y, z, levels, want_fill, want_line)
    fill_data = None
    if coords.fill_coords:
        fill_coords = coords.fill_coords
        fill_data = FillData(xs=fill_coords.xs, ys=fill_coords.ys, lower_levels=levels[:-1], upper_levels=levels[1:])
    line_data = None
    if coords.line_coords:
        line_coords = coords.line_coords
        line_data = LineData(xs=line_coords.xs, ys=line_coords.ys, levels=levels)
    return ContourData(fill_data, line_data)