from __future__ import annotations
import os
import shutil
import warnings
from gzip import GzipFile
from pathlib import Path
from typing import TYPE_CHECKING
from monty.io import zopen
def copy_r(src: str | Path, dst: str | Path) -> None:
    """
    Implements a recursive copy function similar to Unix's "cp -r" command.
    Surprisingly, python does not have a real equivalent. shutil.copytree
    only works if the destination directory is not present.

    Args:
        src (str | Path): Source folder to copy.
        dst (str | Path): Destination folder.
    """
    src = Path(src)
    dst = Path(dst)
    abssrc = src.resolve()
    absdst = dst.resolve()
    os.makedirs(absdst, exist_ok=True)
    for filepath in os.listdir(abssrc):
        fpath = Path(abssrc, filepath)
        if fpath.is_symlink():
            continue
        if fpath.is_file():
            shutil.copy(fpath, absdst)
        elif str(fpath) not in str(absdst):
            copy_r(fpath, Path(absdst, filepath))
        else:
            warnings.warn(f'Cannot copy {fpath} to itself')