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
def path_fallbacks(path: StrPath) -> Generator[str, None, None]:
    """Yield variations of `path` that may exist on the filesystem.

    Return a sequence of paths that should be checked in order for existence or
    create-ability. Essentially, keep replacing "suspect" characters until we run out.
    """
    path = str(path)
    root, tail = os.path.splitdrive(path)
    yield os.path.join(root, tail)
    for char in PROBLEMATIC_PATH_CHARS:
        if char in tail:
            tail = tail.replace(char, '-')
            yield os.path.join(root, tail)