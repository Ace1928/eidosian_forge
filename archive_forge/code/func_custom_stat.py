from __future__ import annotations
import errno
import os
import os.path
import select
import threading
from stat import S_ISDIR
from watchdog.events import (
from watchdog.observers.api import DEFAULT_EMITTER_TIMEOUT, DEFAULT_OBSERVER_TIMEOUT, BaseObserver, EventEmitter
from watchdog.utils import platform
from watchdog.utils.dirsnapshot import DirectorySnapshot
def custom_stat(path, self=self):
    stat_info = stat(path)
    self._register_kevent(path, S_ISDIR(stat_info.st_mode))
    return stat_info