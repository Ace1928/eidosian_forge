import contextlib
import glob
import json
import logging
import os
import platform
import shutil
import tempfile
import traceback
import uuid
from typing import Any, Dict, Iterator, List, Optional, Union
import pyarrow.fs
from ray.air._internal.filelock import TempFileLock
from ray.train._internal.storage import _download_from_fs_path, _exists_at_fs_path
from ray.util.annotations import PublicAPI
@contextlib.contextmanager
def as_directory(self) -> Iterator[str]:
    """Returns checkpoint contents in a local directory as a context.

        This function makes checkpoint data available as a directory while avoiding
        unnecessary copies and left-over temporary data.

        *If the checkpoint points to a local directory*, this method just returns the
        local directory path without making a copy, and nothing will be cleaned up
        after exiting the context.

        *If the checkpoint points to a remote directory*, this method will download the
        checkpoint to a local temporary directory and return the path
        to the temporary directory.

        *If multiple processes on the same node call this method simultaneously,*
        only a single process will perform the download, while the others
        wait for the download to finish. Once the download finishes, all processes
        receive the same local (temporary) directory to read from.

        Once all processes have finished working with the checkpoint,
        the temporary directory is cleaned up.

        Users should treat the returned checkpoint directory as read-only and avoid
        changing any data within it, as it may be deleted when exiting the context.

        Example:

        .. testcode::
            :hide:

            from pathlib import Path
            import tempfile

            from ray.train import Checkpoint

            temp_dir = tempfile.mkdtemp()
            (Path(temp_dir) / "example.txt").write_text("example checkpoint data")
            checkpoint = Checkpoint.from_directory(temp_dir)

        .. testcode::

            with checkpoint.as_directory() as checkpoint_dir:
                # Do some read-only processing of files within checkpoint_dir
                pass

            # At this point, if a temporary directory was created, it will have
            # been deleted.

        """
    if isinstance(self.filesystem, pyarrow.fs.LocalFileSystem):
        yield self.path
    else:
        del_lock_path = _get_del_lock_path(self._get_temporary_checkpoint_dir())
        open(del_lock_path, 'a').close()
        temp_dir = self.to_directory()
        try:
            yield temp_dir
        finally:
            try:
                os.remove(del_lock_path)
            except Exception:
                logger.warning(f'Could not remove {del_lock_path} deletion file lock. Traceback:\n{traceback.format_exc()}')
            remaining_locks = _list_existing_del_locks(temp_dir)
            if not remaining_locks:
                try:
                    with TempFileLock(temp_dir, timeout=0):
                        shutil.rmtree(temp_dir, ignore_errors=True)
                except TimeoutError:
                    pass