from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register_lazy('pyarrow')
def register_pyarrow():
    import pyarrow as pa

    def _get_col_size(data):
        p = 0
        if not isinstance(data, pa.ChunkedArray):
            data = data.data
        for chunk in data.iterchunks():
            for buffer in chunk.buffers():
                if buffer:
                    p += buffer.size
        return p

    @sizeof.register(pa.Table)
    def sizeof_pyarrow_table(table):
        p = sizeof(table.schema.metadata)
        for col in table.itercolumns():
            p += _get_col_size(col)
        return int(p) + 1000

    @sizeof.register(pa.ChunkedArray)
    def sizeof_pyarrow_chunked_array(data):
        return int(_get_col_size(data)) + 1000