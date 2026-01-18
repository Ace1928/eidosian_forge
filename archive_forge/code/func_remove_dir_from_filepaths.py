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
def remove_dir_from_filepaths(base_dir: str, rdir: str):
    """
    base_dir: String path of the directory containing rdir
    rdir: String path of directory relative to base_dir whose contents should
          be moved to its base_dir, its parent directory

    Removes rdir from the filepaths of all files and directories inside it.
    In other words, moves all the files inside rdir to the directory that
    contains rdir. Assumes base_dir's contents and rdir's contents have no
    name conflicts.
    """
    with TemporaryDirectory() as tmp_dir:
        shutil.move(os.path.join(base_dir, rdir), os.path.join(tmp_dir, rdir))
        rdir_children = os.listdir(os.path.join(tmp_dir, rdir))
        for child in rdir_children:
            shutil.move(os.path.join(tmp_dir, rdir, child), os.path.join(base_dir, child))