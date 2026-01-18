from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
def read_long(fo):
    """variable-length, zig-zag encoding."""
    c = fo.read(1)
    b = ord(c)
    n = b & 127
    shift = 7
    while b & 128 != 0:
        b = ord(fo.read(1))
        n |= (b & 127) << shift
        shift += 7
    return n >> 1 ^ -(n & 1)