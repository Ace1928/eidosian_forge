from __future__ import annotations
import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas
import pyarrow as pa
from pandas.core.arrays.arrow.extension_types import ArrowIntervalType
from pandas.core.dtypes.common import _get_dtype, is_string_dtype
from pyarrow.types import is_dictionary
from modin.pandas.indexing import is_range_like
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
@staticmethod
def demangle_index_name(col: str) -> _COL_NAME_TYPE:
    """
        Demangle index column name into index label.

        Parameters
        ----------
        col : str
            Index column name.

        Returns
        -------
        str
            Demangled index name.
        """
    match = ColNameCodec._IDX_NAME_PATTERN.search(col)
    if match:
        name = match.group(1)
        if name == MODIN_UNNAMED_SERIES_LABEL:
            return None
        return ColNameCodec.decode(name)
    return col