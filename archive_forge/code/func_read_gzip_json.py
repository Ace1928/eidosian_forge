from typing import Union, Iterable, Sequence, Any, Optional, Iterator
import sys
import json as _builtin_json
import gzip
from . import ujson
from .util import force_path, force_string, FilePath, JSONInput, JSONOutput
def read_gzip_json(path: FilePath) -> JSONOutput:
    """Load JSON from a gzipped file.

    location (FilePath): The file path.
    RETURNS (JSONOutput): The loaded JSON content.
    """
    file_path = force_string(path)
    with gzip.open(file_path, 'r') as f:
        return ujson.load(f)