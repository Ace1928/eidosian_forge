import contextlib
import ctypes
import errno
import logging
import os
import platform
import re
import shutil
import tempfile
import threading
from pathlib import Path
from typing import IO, Any, BinaryIO, Generator, Optional
from wandb.sdk.lib.paths import StrPath
def reflink(existing_path: StrPath, new_path: StrPath, overwrite: bool=False) -> None:
    """Create a reflink to `existing_path` at `new_path`.

    A reflink (reflective link) is a copy-on-write reference to a file. Once linked, the
    file and link are both "real" files (not symbolic or hard links) and each can be
    modified independently without affecting the other; however, they share the same
    underlying data blocks on disk so until one is modified they are "zero-cost" copies.

    Reflinks have all the functionality of copies, so we should use them wherever they
    are supported if we would otherwise copy a file. (This is not particularly radical--
    GNU `cp` defaults to `reflink=auto`, using it whenever available) However, support
    for them is limited to a small number of filesystems. They should work on:
    - Linux with a Btrfs or XFS filesystem (NOT ext4)
    - macOS 10.13 or later with an APFS filesystem (called clone files)

    Reflinks are also supported on Solaris and Windows with ReFSv2, but we haven't
    implemented support for them.

    Like hard links, a reflink can only be created on the same filesystem as the target.
    """
    if platform.system() == 'Linux':
        link_fn = _reflink_linux
    elif platform.system() == 'Darwin':
        link_fn = _reflink_macos
    else:
        raise OSError(errno.ENOTSUP, f'reflinks are not supported on {platform.system()}')
    new_path = Path(new_path).resolve()
    existing_path = Path(existing_path).resolve()
    if new_path.exists():
        if not overwrite:
            raise FileExistsError(f'{new_path} already exists')
        logger.warning(f'Overwriting existing file {new_path}.')
        new_path.unlink()
    new_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_fn(existing_path, new_path)
    except OSError as e:
        base_msg = f'failed to create reflink from {existing_path} to {new_path}.'
        if e.errno in (errno.EPERM, errno.EACCES):
            raise PermissionError(f'Insufficient permissions; {base_msg}') from e
        if e.errno == errno.ENOENT:
            raise FileNotFoundError(f'File not found; {base_msg}') from e
        if e.errno == errno.EXDEV:
            raise ValueError(f'Cannot link across filesystems; {base_msg}') from e
        if e.errno == errno.EISDIR:
            raise IsADirectoryError(f'Cannot reflink a directory; {base_msg}') from e
        if e.errno in (errno.EOPNOTSUPP, errno.ENOTSUP):
            raise OSError(errno.ENOTSUP, f'Filesystem does not support reflinks; {base_msg}') from e
        if e.errno == errno.EINVAL:
            raise ValueError(f'Cannot link file ranges; {base_msg}') from e
        raise