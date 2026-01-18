from __future__ import annotations
from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import (
import uuid
from pandas._config import using_copy_on_write
from pandas.compat import PYPY
from pandas.errors import ChainedAssignmentError
from pandas import set_option
from pandas.io.common import get_handle
@contextmanager
def decompress_file(path: FilePath | BaseBuffer, compression: CompressionOptions) -> Generator[IO[bytes], None, None]:
    """
    Open a compressed file and return a file object.

    Parameters
    ----------
    path : str
        The path where the file is read from.

    compression : {'gzip', 'bz2', 'zip', 'xz', 'zstd', None}
        Name of the decompression to use

    Returns
    -------
    file object
    """
    with get_handle(path, 'rb', compression=compression, is_text=False) as handle:
        yield handle.handle