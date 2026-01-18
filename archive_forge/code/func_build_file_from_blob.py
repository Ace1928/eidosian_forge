import os
import stat
import struct
import sys
from dataclasses import dataclass
from enum import Enum
from typing import (
from .file import GitFile
from .object_store import iter_tree_contents
from .objects import (
from .pack import ObjectContainer, SHA1Reader, SHA1Writer
def build_file_from_blob(blob: Blob, mode: int, target_path: bytes, *, honor_filemode=True, tree_encoding='utf-8', symlink_fn=None):
    """Build a file or symlink on disk based on a Git object.

    Args:
      blob: The git object
      mode: File mode
      target_path: Path to write to
      honor_filemode: An optional flag to honor core.filemode setting in
        config file, default is core.filemode=True, change executable bit
      symlink: Function to use for creating symlinks
    Returns: stat object for the file
    """
    try:
        oldstat = os.lstat(target_path)
    except FileNotFoundError:
        oldstat = None
    contents = blob.as_raw_string()
    if stat.S_ISLNK(mode):
        if oldstat:
            os.unlink(target_path)
        if sys.platform == 'win32':
            contents = contents.decode(tree_encoding)
            target_path = target_path.decode(tree_encoding)
        (symlink_fn or symlink)(contents, target_path)
    else:
        if oldstat is not None and oldstat.st_size == len(contents):
            with open(target_path, 'rb') as f:
                if f.read() == contents:
                    return oldstat
        with open(target_path, 'wb') as f:
            f.write(contents)
        if honor_filemode:
            os.chmod(target_path, mode)
    return os.lstat(target_path)