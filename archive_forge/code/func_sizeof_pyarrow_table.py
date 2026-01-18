from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(pa.Table)
def sizeof_pyarrow_table(table):
    p = sizeof(table.schema.metadata)
    for col in table.itercolumns():
        p += _get_col_size(col)
    return int(p) + 1000