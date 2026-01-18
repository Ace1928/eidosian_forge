from __future__ import annotations
import os
import shutil
import warnings
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING
from monty.io import zopen
def compress_dir(path: str | Path, compression: Literal['gz', 'bz2']='gz') -> None:
    """
    Recursively compresses all files in a directory. Note that this
    compresses all files singly, i.e., it does not create a tar archive. For
    that, just use Python tarfile class.

    Args:
        path (str | Path): Path to parent directory.
        compression (str): A compression mode. Valid options are "gz" or
            "bz2". Defaults to gz.
    """
    path = Path(path)
    for parent, _, files in os.walk(path):
        for f in files:
            compress_file(Path(parent, f), compression=compression)
    return