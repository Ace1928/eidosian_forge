import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def pandas_ext_num_types(data: DataFrame) -> DataFrame:
    """Experimental suppport for handling pyarrow extension numeric types."""
    import pandas as pd
    import pyarrow as pa
    for col, dtype in zip(data.columns, data.dtypes):
        if not is_pa_ext_dtype(dtype):
            continue
        d_array: pd.arrays.ArrowExtensionArray = data[col].array
        aa: pa.ChunkedArray = d_array.__arrow_array__()
        chunk: pa.Array = aa.combine_chunks()
        arr = chunk.__array__()
        data[col] = arr
    return data