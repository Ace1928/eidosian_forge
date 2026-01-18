from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def bool_column_to_array(col: ColumnObject, allow_copy: bool=True) -> pa.Array:
    """
    Convert a column holding boolean dtype to a PyArrow array.

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
    size = buffers['data'][1][1]
    if size == 8 and (not allow_copy):
        raise RuntimeError('Boolean column will be casted from uint8 and a copy is required which is forbidden by allow_copy=False')
    data_type = col.dtype
    data = buffers_to_array(buffers, data_type, col.size(), col.describe_null, col.offset)
    if size == 8:
        data = pc.cast(data, pa.bool_())
    return data