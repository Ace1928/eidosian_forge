from __future__ import annotations
import logging # isort:skip
from typing import (
import numpy as np
from ..core.has_props import abstract
from ..core.properties import (
from ..model import Model
from ..util.deprecation import deprecated
from ..util.serialization import convert_datetime_array
from ..util.warnings import BokehUserWarning, warn
from .callbacks import CustomJS
from .filters import AllIndices, Filter, IntersectionFilter
from .selections import Selection, SelectionPolicy, UnionRenderers
@classmethod
def from_groupby(cls, data: pd.core.groupby.GroupBy) -> DataDict:
    """ Create a ``dict`` of columns from a Pandas ``GroupBy``,
        suitable for creating a ``ColumnDataSource``.

        The data generated is the result of running ``describe``
        on the group.

        Args:
            data (Groupby) : data to convert

        Returns:
            dict[str, np.array]

        """
    return cls._data_from_df(data.describe())