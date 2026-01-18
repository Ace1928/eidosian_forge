from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Sequence, Union
from .core.property.vectorization import Expr, Field
from .models.expressions import CumSum, Stack
from .models.mappers import (
from .models.transforms import Dodge, Jitter
def dodge(field_name: str, value: float, range: Range | None=None) -> Field:
    """ Create a ``DataSpec`` dict that applies a client-side ``Dodge``
    transformation to a ``ColumnDataSource`` column.

    Args:
        field_name (str) : a field name to configure ``DataSpec`` with

        value (float) : the fixed offset to add to column data

        range (Range, optional) : a range to use for computing synthetic
            coordinates when necessary, e.g. a ``FactorRange`` when the
            column data is categorical (default: None)

    Returns:
        Field

    """
    return Field(field_name, Dodge(value=value, range=range))