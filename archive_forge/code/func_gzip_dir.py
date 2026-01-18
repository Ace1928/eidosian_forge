from __future__ import annotations
import os
import shutil
import warnings
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING
from monty.io import zopen
def gzip_dir(path: str | Path, compresslevel: int=6) -> None:
    """
    Gzips all files in a directory. Note that this is different from
    shutil.make_archive, which creates a tar archive. The aim of this method
    is to create gzipped files that can still be read using common Unix-style
    commands like zless or zcat.

    Args:
        path (str | Path): Path to directory.
        compresslevel (int): Level of compression, 1-9. 9 is default for
            GzipFile, 6 is default for gzip.
    """
    path = Path(path)
    for root, _, files in os.walk(path):
        for f in files:
            full_f = Path(root, f).resolve()
            if Path(f).suffix.lower() != '.gz' and (not full_f.is_dir()):
                with open(full_f, 'rb') as f_in, GzipFile(f'{full_f}.gz', 'wb', compresslevel=compresslevel) as f_out:
                    shutil.copyfileobj(f_in, f_out)
                shutil.copystat(full_f, f'{full_f}.gz')
                os.remove(full_f)