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
@property
def checkpoint_fs_path(self) -> str:
    """The current checkpoint directory path on the `storage_filesystem`.

        "Current" refers to the checkpoint that is currently being created/persisted.
        The user of this class is responsible for setting the `current_checkpoint_index`
        (e.g., incrementing when needed).
        """
    return Path(self.trial_fs_path, self.checkpoint_dir_name).as_posix()