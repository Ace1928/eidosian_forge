import concurrent.futures
import logging
import os
import queue
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Optional, Tuple
import wandb
import wandb.util
from wandb.filesync import stats, step_checksum, step_upload
from wandb.sdk.lib.paths import LogicalPath
def file_changed(self, save_name: LogicalPath, path: str, copy: bool=True):
    """Tell the file pusher that a file's changed and should be uploaded.

        Arguments:
            save_name: string logical location of the file relative to the run
                directory.
            path: actual string path of the file to upload on the filesystem.
        """
    if not os.path.exists(path) or not os.path.isfile(path):
        return
    if os.path.getsize(path) == 0:
        return
    event = step_checksum.RequestUpload(path, save_name, copy)
    self._incoming_queue.put(event)