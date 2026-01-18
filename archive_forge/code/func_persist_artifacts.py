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
def persist_artifacts(self, force: bool=False) -> None:
    """Persists all artifacts within `trial_local_dir` to storage.

        This method possibly launches a background task to sync the trial dir,
        depending on the `sync_period` + `sync_artifacts_on_checkpoint`
        settings of `SyncConfig`.

        `(local_fs, trial_local_path) -> (storage_filesystem, trial_fs_path)`

        Args:
            force: If True, wait for a previous sync to finish, launch a new one,
                and wait for that one to finish. By the end of a `force=True` call, the
                latest version of the trial artifacts will be persisted.
        """
    if not self.sync_config.sync_artifacts:
        return
    if not self.syncer:
        return
    if force:
        self.syncer.wait()
        self.syncer.sync_up(local_dir=self.trial_local_path, remote_dir=self.trial_fs_path)
        self.syncer.wait()
    else:
        self.syncer.sync_up_if_needed(local_dir=self.trial_local_path, remote_dir=self.trial_fs_path)