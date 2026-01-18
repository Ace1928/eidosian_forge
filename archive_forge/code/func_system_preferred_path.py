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
def system_preferred_path(path: StrPath, warn: bool=False) -> StrPath:
    """Replace ':' with '-' in paths on Windows.

    Args:
        path: The path to convert.
        warn: Whether to warn if ':' is replaced.
    """
    if platform.system() != 'Windows':
        return path
    head, tail = os.path.splitdrive(path)
    if warn and ':' in tail:
        logger.warning(f"Replacing ':' in {tail} with '-'")
    new_path = head + tail.replace(':', '-')
    return Path(new_path) if isinstance(path, Path) else new_path