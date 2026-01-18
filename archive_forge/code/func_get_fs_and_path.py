import dataclasses
import fnmatch
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union
from ray._private.storage import _get_storage_uri
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.syncer import SyncConfig, Syncer, _BackgroundSyncer
from ray.train.constants import _get_defaults_results_dir
def get_fs_and_path(storage_path: Union[str, os.PathLike], storage_filesystem: Optional[pyarrow.fs.FileSystem]=None) -> Tuple[pyarrow.fs.FileSystem, str]:
    """Returns the fs and path from a storage path and an optional custom fs.

    Args:
        storage_path: A storage path or URI. (ex: s3://bucket/path or /tmp/ray_results)
        storage_filesystem: A custom filesystem to use. If not provided,
            this will be auto-resolved by pyarrow. If provided, the storage_path
            is assumed to be prefix-stripped already, and must be a valid path
            on the filesystem.
    """
    storage_path = str(storage_path)
    if storage_filesystem:
        return (storage_filesystem, storage_path)
    return pyarrow.fs.FileSystem.from_uri(storage_path)