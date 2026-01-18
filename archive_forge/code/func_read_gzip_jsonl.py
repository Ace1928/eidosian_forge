from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def read_gzip_jsonl(path: FilePath, skip: bool=False) -> Iterator[JSONOutput]:
    """Read a gzipped .jsonl file and yield contents line by line.
    Blank lines will always be skipped.

    path (FilePath): The file path.
    skip (bool): Skip broken lines and don't raise ValueError.
    YIELDS (JSONOutput): The unpacked, deserialized Python objects.
    """
    with gzip.open(force_path(path), 'r') as f:
        for line in _yield_json_lines(f, skip=skip):
            yield line