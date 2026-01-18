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
def table_to_dataframe(options, table, categories=None, ignore_metadata=False, types_mapper=None):
    from pandas.core.internals import BlockManager
    from pandas import DataFrame
    all_columns = []
    column_indexes = []
    pandas_metadata = table.schema.pandas_metadata
    if not ignore_metadata and pandas_metadata is not None:
        all_columns = pandas_metadata['columns']
        column_indexes = pandas_metadata.get('column_indexes', [])
        index_descriptors = pandas_metadata['index_columns']
        table = _add_any_metadata(table, pandas_metadata)
        table, index = _reconstruct_index(table, index_descriptors, all_columns, types_mapper)
        ext_columns_dtypes = _get_extension_dtypes(table, all_columns, types_mapper)
    else:
        index = _pandas_api.pd.RangeIndex(table.num_rows)
        ext_columns_dtypes = _get_extension_dtypes(table, [], types_mapper)
    _check_data_column_metadata_consistency(all_columns)
    columns = _deserialize_column_index(table, all_columns, column_indexes)
    blocks = _table_to_blocks(options, table, categories, ext_columns_dtypes)
    axes = [columns, index]
    mgr = BlockManager(blocks, axes)
    if _pandas_api.is_ge_v21():
        df = DataFrame._from_mgr(mgr, mgr.axes)
    else:
        df = DataFrame(mgr)
    return df