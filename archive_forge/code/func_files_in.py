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
def files_in(path: StrPath) -> Generator[os.DirEntry, None, None]:
    """Yield a directory entry for each file under a given path (recursive)."""
    if not os.path.isdir(path):
        return
    for entry in os.scandir(path):
        if entry.is_dir():
            yield from files_in(entry.path)
        else:
            yield entry