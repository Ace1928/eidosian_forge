from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence, Union
from .core.property.vectorization import Expr, Field
from .models.expressions import CumSum, Stack
from .models.mappers import (
from .models.transforms import Dodge, Jitter
def factor_cmap(field_name: str, palette: Sequence[ColorLike], factors: Factors, start: float=0, end: float | None=None, nan_color: ColorLike='gray') -> Field:
    """ Create a ``DataSpec`` dict that applies a client-side
    ``CategoricalColorMapper`` transformation to a ``ColumnDataSource``
    column.

    Args:
        field_name (str) : a field name to configure ``DataSpec`` with

        palette (seq[color]) : a list of colors to use for colormapping

        factors (seq) : a sequence of categorical factors corresponding to
            the palette

        start (int, optional) : a start slice index to apply when the column
            data has factors with multiple levels. (default: 0)

        end (int, optional) : an end slice index to apply when the column
            data has factors with multiple levels. (default: None)

        nan_color (color, optional) : a default color to use when mapping data
            from a column does not succeed (default: "gray")

    Returns:
        Field

    """
    return Field(field_name, CategoricalColorMapper(palette=palette, factors=factors, start=start, end=end, nan_color=nan_color))