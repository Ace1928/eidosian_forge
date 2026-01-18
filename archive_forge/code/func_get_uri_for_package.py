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
def get_uri_for_package(package: Path) -> str:
    """Get a content-addressable URI from a package's contents."""
    if package.suffix == '.whl':
        return '{protocol}://{whl_filename}'.format(protocol=Protocol.GCS.value, whl_filename=package.name)
    else:
        hash_val = hashlib.md5(package.read_bytes()).hexdigest()
        return '{protocol}://{pkg_name}.zip'.format(protocol=Protocol.GCS.value, pkg_name=RAY_PKG_PREFIX + hash_val)