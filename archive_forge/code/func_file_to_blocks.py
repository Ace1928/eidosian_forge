from __future__ import annotations
import io
from functools import partial
from fsspec.core import open_files
from tlz import concat
from dask.bag.core import from_delayed
from dask.bytes import read_bytes
from dask.delayed import delayed
from dask.utils import parse_bytes, system_encoding
def file_to_blocks(include_path, lazy_file, delimiter=None):
    with lazy_file as f:
        if delimiter is not None:
            text = f.read()
            if not text:
                return []
            parts = text.split(delimiter)
            yield from ((line, lazy_file.path) if include_path else line for line in [line + delimiter for line in parts[:-1]] + parts[-1:])
        else:
            for line in f:
                yield ((line, lazy_file.path) if include_path else line)