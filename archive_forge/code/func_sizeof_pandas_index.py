from __future__ import annotations
import itertools
import logging
import random
import sys
from array import array
from packaging.version import parse as parse_version
from dask._compatibility import importlib_metadata
from dask.utils import Dispatch
@sizeof.register(pd.Index)
def sizeof_pandas_index(i):
    p = 400 + i.memory_usage(deep=False)
    if i.dtype in OBJECT_DTYPES:
        p += object_size(i)
    return p