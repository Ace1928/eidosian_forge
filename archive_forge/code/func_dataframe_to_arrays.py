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
def dataframe_to_arrays(df, schema, preserve_index, nthreads=1, columns=None, safe=True):
    all_names, column_names, index_column_names, index_descriptors, index_columns, columns_to_convert, convert_fields = _get_columns_to_convert(df, schema, preserve_index, columns)
    if nthreads is None:
        nrows, ncols = (len(df), len(df.columns))
        if nrows > ncols * 100 and ncols > 1:
            nthreads = pa.cpu_count()
        else:
            nthreads = 1

    def convert_column(col, field):
        if field is None:
            field_nullable = True
            type_ = None
        else:
            field_nullable = field.nullable
            type_ = field.type
        try:
            result = pa.array(col, type=type_, from_pandas=True, safe=safe)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError) as e:
            e.args += ('Conversion failed for column {!s} with type {!s}'.format(col.name, col.dtype),)
            raise e
        if not field_nullable and result.null_count > 0:
            raise ValueError('Field {} was non-nullable but pandas column had {} null values'.format(str(field), result.null_count))
        return result

    def _can_definitely_zero_copy(arr):
        return isinstance(arr, np.ndarray) and arr.flags.contiguous and issubclass(arr.dtype.type, np.integer)
    if nthreads == 1:
        arrays = [convert_column(c, f) for c, f in zip(columns_to_convert, convert_fields)]
    else:
        arrays = []
        with futures.ThreadPoolExecutor(nthreads) as executor:
            for c, f in zip(columns_to_convert, convert_fields):
                if _can_definitely_zero_copy(c.values):
                    arrays.append(convert_column(c, f))
                else:
                    arrays.append(executor.submit(convert_column, c, f))
        for i, maybe_fut in enumerate(arrays):
            if isinstance(maybe_fut, futures.Future):
                arrays[i] = maybe_fut.result()
    types = [x.type for x in arrays]
    if schema is None:
        fields = []
        for name, type_ in zip(all_names, types):
            name = name if name is not None else 'None'
            fields.append(pa.field(name, type_))
        schema = pa.schema(fields)
    pandas_metadata = construct_metadata(columns_to_convert, df, column_names, index_columns, index_descriptors, preserve_index, types)
    metadata = deepcopy(schema.metadata) if schema.metadata else dict()
    metadata.update(pandas_metadata)
    schema = schema.with_metadata(metadata)
    n_rows = None
    if len(arrays) == 0:
        try:
            kind = index_descriptors[0]['kind']
            if kind == 'range':
                start = index_descriptors[0]['start']
                stop = index_descriptors[0]['stop']
                step = index_descriptors[0]['step']
                n_rows = len(range(start, stop, step))
        except IndexError:
            pass
    return (arrays, schema, n_rows)