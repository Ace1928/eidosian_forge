from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def column_to_array(col: ColumnObject, allow_copy: bool=True) -> pa.Array:
    """
    Convert a column holding one of the primitive dtypes to a PyArrow array.
    A primitive type is one of: int, uint, float, bool (1 bit).

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array
    """
    buffers = col.get_buffers()
    data_type = col.dtype
    data = buffers_to_array(buffers, data_type, col.size(), col.describe_null, col.offset, allow_copy)
    return data