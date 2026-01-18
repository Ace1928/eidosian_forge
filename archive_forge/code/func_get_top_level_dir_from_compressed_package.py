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
def get_top_level_dir_from_compressed_package(package_path: str):
    """
    If compressed package at package_path contains a single top-level
    directory, returns the name of the top-level directory. Otherwise,
    returns None.

    Ignores a second top-level directory if it is named __MACOSX.
    """
    package_zip = ZipFile(package_path, 'r')
    top_level_directory = None

    def is_top_level_file(file_name):
        return '/' not in file_name

    def base_dir_name(file_name):
        return file_name.split('/')[0]
    for file_name in package_zip.namelist():
        if top_level_directory is None:
            if is_top_level_file(file_name):
                return None
            else:
                dir_name = base_dir_name(file_name)
                if dir_name == MAC_OS_ZIP_HIDDEN_DIR_NAME:
                    continue
                top_level_directory = dir_name
        elif is_top_level_file(file_name) or base_dir_name(file_name) not in [top_level_directory, MAC_OS_ZIP_HIDDEN_DIR_NAME]:
            return None
    return top_level_directory