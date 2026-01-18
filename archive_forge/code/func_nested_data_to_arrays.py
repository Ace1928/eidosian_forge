from __future__ import annotations
from collections import abc
from typing import (
import numpy as np
from numpy import ma
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.core.dtypes.astype import astype_is_view
from pandas.core.dtypes.cast import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
from pandas.core import (
from pandas.core.arrays import ExtensionArray
from pandas.core.arrays.string_ import StringDtype
from pandas.core.construction import (
from pandas.core.indexes.api import (
from pandas.core.internals.array_manager import (
from pandas.core.internals.blocks import (
from pandas.core.internals.managers import (
def nested_data_to_arrays(data: Sequence, columns: Index | None, index: Index | None, dtype: DtypeObj | None) -> tuple[list[ArrayLike], Index, Index]:
    """
    Convert a single sequence of arrays to multiple arrays.
    """
    if is_named_tuple(data[0]) and columns is None:
        columns = ensure_index(data[0]._fields)
    arrays, columns = to_arrays(data, columns, dtype=dtype)
    columns = ensure_index(columns)
    if index is None:
        if isinstance(data[0], ABCSeries):
            index = _get_names_from_index(data)
        else:
            index = default_index(len(data))
    return (arrays, columns, index)