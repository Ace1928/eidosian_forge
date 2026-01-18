from __future__ import annotations
import os
import stat
import tarfile
import tempfile
import time
import typing as t
from .constants import (
from .config import (
from .util import (
from .data import (
from .util_common import (
def apply_permissions(tar_info: tarfile.TarInfo, mode: int) -> t.Optional[tarfile.TarInfo]:
    """
        Apply the specified permissions to the given file.
        Existing file type bits are preserved.
        """
    tar_info.mode &= ~(stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    tar_info.mode |= mode
    return tar_info