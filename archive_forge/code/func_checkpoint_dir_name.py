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
def checkpoint_dir_name(self) -> str:
    """The current checkpoint directory name, based on the checkpoint index."""
    return StorageContext._make_checkpoint_dir_name(self.current_checkpoint_index)