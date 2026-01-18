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
def persist_current_checkpoint(self, checkpoint: 'Checkpoint') -> 'Checkpoint':
    """Persists a given checkpoint to the current checkpoint path on the filesystem.

        "Current" is defined by the `current_checkpoint_index` attribute of the
        storage context.

        This method copies the checkpoint files to the storage location.
        It's up to the user to delete the original checkpoint files if desired.

        For example, the original directory is typically a local temp directory.

        Args:
            checkpoint: The checkpoint to persist to (fs, checkpoint_fs_path).

        Returns:
            Checkpoint: A Checkpoint pointing to the persisted checkpoint location.
        """
    from ray.train._checkpoint import Checkpoint
    logger.debug('Copying checkpoint files to storage path:\n({source_fs}, {source}) -> ({dest_fs}, {destination})'.format(source=checkpoint.path, destination=self.checkpoint_fs_path, source_fs=checkpoint.filesystem, dest_fs=self.storage_filesystem))
    self._check_validation_file()
    self.storage_filesystem.create_dir(self.checkpoint_fs_path)
    _pyarrow_fs_copy_files(source=checkpoint.path, destination=self.checkpoint_fs_path, source_filesystem=checkpoint.filesystem, destination_filesystem=self.storage_filesystem)
    persisted_checkpoint = Checkpoint(filesystem=self.storage_filesystem, path=self.checkpoint_fs_path)
    logger.info(f'Checkpoint successfully created at: {persisted_checkpoint}')
    return persisted_checkpoint