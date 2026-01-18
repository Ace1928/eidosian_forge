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
def is_supported_arrow_type(dtype: pa.lib.DataType) -> bool:
    """
    Return True if the specified arrow type is supported by HDK.

    Parameters
    ----------
    dtype : pa.lib.DataType

    Returns
    -------
    bool
    """
    if pa.types.is_string(dtype) or pa.types.is_time(dtype) or pa.types.is_dictionary(dtype) or pa.types.is_null(dtype):
        return True
    if isinstance(dtype, pa.ExtensionType) or pa.types.is_duration(dtype):
        return False
    try:
        pandas_dtype = dtype.to_pandas_dtype()
        return pandas_dtype != pandas.api.types.pandas_dtype('O')
    except NotImplementedError:
        return False