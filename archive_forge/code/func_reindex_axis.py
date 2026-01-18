from __future__ import annotations
from typing import (
import numpy as np
from pandas._config import (
from pandas._libs import (
from pandas.errors import AbstractMethodError
from pandas.util._validators import validate_bool_kwarg
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.dtypes import (
from pandas.core.base import PandasObject
from pandas.core.construction import extract_array
from pandas.core.indexes.api import (
@final
def reindex_axis(self, new_index: Index, axis: AxisInt, fill_value=None, only_slice: bool=False) -> Self:
    """
        Conform data manager to new index.
        """
    new_index, indexer = self.axes[axis].reindex(new_index)
    return self.reindex_indexer(new_index, indexer, axis=axis, fill_value=fill_value, copy=False, only_slice=only_slice)