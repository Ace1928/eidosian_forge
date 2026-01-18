from collections.abc import Mapping, MutableMapping
from functools import partial
from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, TypeVar, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from packaging import version
from .. import config
from ..features import Features
from ..features.features import _ArrayXDExtensionType, _is_zero_copy_only, decode_nested_example, pandas_types_mapper
from ..table import Table
from ..utils.py_utils import no_op_if_value_is_null
def query_table(table: Table, key: Union[int, slice, range, str, Iterable], indices: Optional[Table]=None) -> pa.Table:
    """
    Query a Table to extract the subtable that correspond to the given key.

    Args:
        table (``datasets.table.Table``): The input Table to query from
        key (``Union[int, slice, range, str, Iterable]``): The key can be of different types:
            - an integer i: the subtable containing only the i-th row
            - a slice [i:j:k]: the subtable containing the rows that correspond to this slice
            - a range(i, j, k): the subtable containing the rows that correspond to this range
            - a string c: the subtable containing all the rows but only the column c
            - an iterable l: the subtable that is the concatenation of all the i-th rows for all i in the iterable
        indices (Optional ``datasets.table.Table``): If not None, it is used to re-map the given key to the table rows.
            The indices table must contain one column named "indices" of type uint64.
            This is used in case of shuffling or rows selection.


    Returns:
        ``pyarrow.Table``: the result of the query on the input table
    """
    if not isinstance(key, (int, slice, range, str, Iterable)):
        _raise_bad_key_type(key)
    if isinstance(key, str):
        _check_valid_column_key(key, table.column_names)
    else:
        size = indices.num_rows if indices is not None else table.num_rows
        _check_valid_index_key(key, size)
    if indices is None:
        pa_subtable = _query_table(table, key)
    else:
        pa_subtable = _query_table_with_indices_mapping(table, key, indices=indices)
    return pa_subtable