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
def arrow_type_to_pandas(at: pa.lib.DataType):
    """
    Convert the specified arrow type to pandas dtype.

    Parameters
    ----------
    at : pa.lib.DataType

    Returns
    -------
    dtype
    """
    if at == pa.string():
        return _get_dtype(str)
    return at.to_pandas_dtype()