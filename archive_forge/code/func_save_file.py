import abc
import fnmatch
import glob
import logging
import os
import queue
import time
from typing import TYPE_CHECKING, Any, Mapping, MutableMapping, MutableSet, Optional
from wandb import util
from wandb.sdk.interface.interface import GlobStr
from wandb.sdk.lib.paths import LogicalPath
def save_file(self) -> None:
    self._last_sync = os.path.getmtime(self.file_path)
    self._last_uploaded_time = time.time()
    self._last_uploaded_size = self.current_size
    self._file_pusher.file_changed(self.save_name, self.file_path)