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
def check_exists(path: StrPath) -> Optional[StrPath]:
    """Look for variations of `path` and return the first found.

    This exists to support former behavior around system-dependent paths; we used to use
    ':' in Artifact paths unless we were on Windows, but this has issues when e.g. a
    Linux machine is accessing an NTFS filesystem; we might need to look for the
    alternate path. This checks all the possible directories we would consider creating.
    """
    for dest in path_fallbacks(path):
        if os.path.exists(dest):
            return Path(dest) if isinstance(path, Path) else dest
    return None