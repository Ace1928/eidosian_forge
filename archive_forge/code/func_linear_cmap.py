from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence, Union
from .core.property.vectorization import Expr, Field
from .models.expressions import CumSum, Stack
from .models.mappers import (
from .models.transforms import Dodge, Jitter
def linear_cmap(field_name: str, palette: Sequence[ColorLike], low: float, high: float, low_color: ColorLike | None=None, high_color: ColorLike | None=None, nan_color: ColorLike='gray') -> Field:
    """ Create a ``DataSpec`` dict that applies a client-side
    ``LinearColorMapper`` transformation to a ``ColumnDataSource`` column.

    Args:
        field_name (str) : a field name to configure ``DataSpec`` with

        palette (seq[color]) : a list of colors to use for colormapping

        low (float) : a minimum value of the range to map into the palette.
            Values below this are clamped to ``low``.

        high (float) : a maximum value of the range to map into the palette.
            Values above this are clamped to ``high``.

        low_color (color, optional) : color to be used if data is lower than
            ``low`` value. If None, values lower than ``low`` are mapped to the
            first color in the palette. (default: None)

        high_color (color, optional) : color to be used if data is higher than
            ``high`` value. If None, values higher than ``high`` are mapped to
            the last color in the palette. (default: None)

        nan_color (color, optional) : a default color to use when mapping data
            from a column does not succeed (default: "gray")

    """
    return Field(field_name, LinearColorMapper(palette=palette, low=low, high=high, nan_color=nan_color, low_color=low_color, high_color=high_color))