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
def ensure_supported_dtype(dtype):
    """
    Check if the specified `dtype` is supported by HDK.

    If `dtype` is not supported, `NotImplementedError` is raised.

    Parameters
    ----------
    dtype : dtype
    """
    try:
        dtype = pa.from_numpy_dtype(dtype)
    except pa.ArrowNotImplementedError as err:
        raise NotImplementedError(f'Type {dtype}') from err
    if not is_supported_arrow_type(dtype):
        raise NotImplementedError(f'Type {dtype}')