import asyncio
import hashlib
import logging
import os
import shutil
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional, Tuple
from urllib.parse import urlparse
from zipfile import ZipFile
from filelock import FileLock
from ray.util.annotations import DeveloperAPI
from ray._private.ray_constants import (
from ray._private.runtime_env.conda_utils import exec_cmd_stream_to_logger
from ray._private.thirdparty.pathspec import PathSpec
from ray.experimental.internal_kv import (
def delete_package(pkg_uri: str, base_directory: str) -> Tuple[bool, int]:
    """Deletes a specific URI from the local filesystem.

    Args:
        pkg_uri: URI to delete.

    Returns:
        bool: True if the URI was successfully deleted, else False.
    """
    deleted = False
    path = Path(_get_local_path(base_directory, pkg_uri))
    with FileLock(str(path) + '.lock'):
        path = path.with_suffix('')
        if path.exists():
            if path.is_dir() and (not path.is_symlink()):
                shutil.rmtree(str(path))
            else:
                path.unlink()
            deleted = True
    return deleted