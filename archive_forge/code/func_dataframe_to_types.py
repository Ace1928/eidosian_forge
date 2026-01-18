import ast
from collections.abc import Sequence
from concurrent import futures
import concurrent.futures.thread  # noqa
from copy import deepcopy
from itertools import zip_longest
import json
import operator
import re
import warnings
import numpy as np
import pyarrow as pa
from pyarrow.lib import _pandas_api, frombytes  # noqa
def dataframe_to_types(df, preserve_index, columns=None):
    all_names, column_names, _, index_descriptors, index_columns, columns_to_convert, _ = _get_columns_to_convert(df, None, preserve_index, columns)
    types = []
    for c in columns_to_convert:
        values = c.values
        if _pandas_api.is_categorical(values):
            type_ = pa.array(c, from_pandas=True).type
        elif _pandas_api.is_extension_array_dtype(values):
            empty = c.head(0) if isinstance(c, _pandas_api.pd.Series) else c[:0]
            type_ = pa.array(empty, from_pandas=True).type
        else:
            values, type_ = get_datetimetz_type(values, c.dtype, None)
            type_ = pa.lib._ndarray_to_arrow_type(values, type_)
            if type_ is None:
                type_ = pa.array(c, from_pandas=True).type
        types.append(type_)
    metadata = construct_metadata(columns_to_convert, df, column_names, index_columns, index_descriptors, preserve_index, types)
    return (all_names, types, metadata)