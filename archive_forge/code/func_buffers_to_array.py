from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def buffers_to_array(buffers: ColumnBuffers, data_type: Tuple[DtypeKind, int, str, str], length: int, describe_null: ColumnNullType, offset: int=0, allow_copy: bool=True) -> pa.Array:
    """
    Build a PyArrow array from the passed buffer.

    Parameters
    ----------
    buffer : ColumnBuffers
        Dictionary containing tuples of underlying buffers and
        their associated dtype.
    data_type : Tuple[DtypeKind, int, str, str],
        Dtype description of the column as a tuple ``(kind, bit-width, format string,
        endianness)``.
    length : int
        The number of values in the array.
    describe_null: ColumnNullType
        Null representation the column dtype uses,
        as a tuple ``(kind, value)``
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.Array

    Notes
    -----
    The returned array doesn't own the memory. The caller of this function
    is responsible for keeping the memory owner object alive as long as
    the returned PyArrow array is being used.
    """
    data_buff, _ = buffers['data']
    try:
        validity_buff, validity_dtype = buffers['validity']
    except TypeError:
        validity_buff = None
    try:
        offset_buff, offset_dtype = buffers['offsets']
    except TypeError:
        offset_buff = None
    data_pa_buffer = pa.foreign_buffer(data_buff.ptr, data_buff.bufsize, base=data_buff)
    if validity_buff:
        validity_pa_buff = validity_buffer_from_mask(validity_buff, validity_dtype, describe_null, length, offset, allow_copy)
    else:
        validity_pa_buff = validity_buffer_nan_sentinel(data_pa_buffer, data_type, describe_null, length, offset, allow_copy)
    data_dtype = map_date_type(data_type)
    if offset_buff:
        _, offset_bit_width, _, _ = offset_dtype
        offset_pa_buffer = pa.foreign_buffer(offset_buff.ptr, offset_buff.bufsize, base=offset_buff)
        if data_type[2] == 'U':
            string_type = pa.large_string()
        elif offset_bit_width == 64:
            string_type = pa.large_string()
        else:
            string_type = pa.string()
        array = pa.Array.from_buffers(string_type, length, [validity_pa_buff, offset_pa_buffer, data_pa_buffer], offset=offset)
    else:
        array = pa.Array.from_buffers(data_dtype, length, [validity_pa_buff, data_pa_buffer], offset=offset)
    return array