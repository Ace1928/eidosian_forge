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
def detect_permissions(tar_info: tarfile.TarInfo) -> t.Optional[tarfile.TarInfo]:
    """
        Detect and apply the appropriate permissions for a file.
        Existing file type bits are preserved.
        This ensures consistency of test results when using unprivileged users.
        """
    if tar_info.path.startswith('ansible/'):
        mode = permissions.get(os.path.relpath(tar_info.path, 'ansible'))
    elif data_context().content.collection and is_subdir(tar_info.path, data_context().content.collection.directory):
        mode = permissions.get(os.path.relpath(tar_info.path, data_context().content.collection.directory))
    else:
        mode = None
    if mode:
        tar_info = apply_permissions(tar_info, mode)
    elif tar_info.mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
        tar_info = make_executable(tar_info)
    else:
        tar_info = make_non_executable(tar_info)
    return tar_info