from __future__ import annotations
from typing import (
from pyarrow.interchange.column import (
import pyarrow as pa
import re
import pyarrow.compute as pc
from pyarrow.interchange.column import Dtype
def categorical_column_to_dictionary(col: ColumnObject, allow_copy: bool=True) -> pa.DictionaryArray:
    """
    Convert a column holding categorical data to a pa.DictionaryArray.

    Parameters
    ----------
    col : ColumnObject
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pa.DictionaryArray
    """
    if not allow_copy:
        raise RuntimeError('Categorical column will be casted from uint8 and a copy is required which is forbidden by allow_copy=False')
    categorical = col.describe_categorical
    if not categorical['is_dictionary']:
        raise NotImplementedError('Non-dictionary categoricals not supported yet')
    cat_column = categorical['categories']
    dictionary = column_to_array(cat_column)
    buffers = col.get_buffers()
    _, data_type = buffers['data']
    indices = buffers_to_array(buffers, data_type, col.size(), col.describe_null, col.offset)
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    return dict_array