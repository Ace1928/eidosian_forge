from __future__ import annotations
import io
import logging
import os
import re
from glob import has_magic
from pathlib import Path
from .caching import (  # noqa: F401
from .compression import compr
from .registry import filesystem, get_filesystem_class
from .utils import (
def open_local(url: str | list[str] | Path | list[Path], mode: str='rb', **storage_options: dict) -> str | list[str]:
    """Open file(s) which can be resolved to local

    For files which either are local, or get downloaded upon open
    (e.g., by file caching)

    Parameters
    ----------
    url: str or list(str)
    mode: str
        Must be read mode
    storage_options:
        passed on to FS for or used by open_files (e.g., compression)
    """
    if 'r' not in mode:
        raise ValueError('Can only ensure local files when reading')
    of = open_files(url, mode=mode, **storage_options)
    if not getattr(of[0].fs, 'local_file', False):
        raise ValueError('open_local can only be used on a filesystem which has attribute local_file=True')
    with of as files:
        paths = [f.name for f in files]
    if isinstance(url, str) and (not has_magic(url)) or isinstance(url, Path):
        return paths[0]
    return paths