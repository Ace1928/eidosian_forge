from __future__ import annotations
import io
import uuid
from fsspec.core import OpenFile, get_fs_token_paths, open_files
from fsspec.utils import read_block
from fsspec.utils import tokenize as fs_tokenize
from dask.highlevelgraph import HighLevelGraph
def open_head(fs, path, compression):
    """Open a file just to read its head and size"""
    with OpenFile(fs, path, compression=compression) as f:
        head = read_header(f)
    size = fs.info(path)['size']
    return (head, size)