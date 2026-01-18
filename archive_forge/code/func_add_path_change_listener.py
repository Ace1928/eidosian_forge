from __future__ import annotations
import os
import threading
from typing import Callable, Final, cast
from blinker import ANY, Signal
from watchdog import events
from watchdog.observers import Observer
from watchdog.observers.api import ObservedWatch
from streamlit.logger import get_logger
from streamlit.util import repr_
from streamlit.watcher import util
def add_path_change_listener(self, path: str, callback: Callable[[str], None], *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> None:
    """Add a path to this object's event filter."""
    with self._lock:
        watched_path = self._watched_paths.get(path, None)
        if watched_path is None:
            md5 = util.calc_md5_with_blocking_retries(path, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
            modification_time = util.path_modification_time(path, allow_nonexistent)
            watched_path = WatchedPath(md5=md5, modification_time=modification_time, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)
            self._watched_paths[path] = watched_path
        watched_path.on_changed.connect(callback, weak=False)