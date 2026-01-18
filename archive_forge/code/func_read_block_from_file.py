from __future__ import annotations
import copy
import os
from fsspec.core import OpenFile, get_fs_token_paths
from fsspec.utils import infer_compression, read_block
from dask.base import tokenize
from dask.delayed import delayed
from dask.utils import is_integer, parse_bytes
def read_block_from_file(lazy_file, off, bs, delimiter):
    with copy.copy(lazy_file) as f:
        if off == 0 and bs is None:
            return f.read()
        return read_block(f, off, bs, delimiter)