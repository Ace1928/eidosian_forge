from __future__ import annotations
import io
from functools import partial
from fsspec.core import open_files
from tlz import concat
from dask.bag.core import from_delayed
from dask.bytes import read_bytes
from dask.delayed import delayed
from dask.utils import parse_bytes, system_encoding
def attach_path(block, path):
    for p in block:
        yield (p, path)