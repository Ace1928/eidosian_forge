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
def stop_watching_path(self, path: str, callback: Callable[[str], None]) -> None:
    """Stop watching a path."""
    folder_path = os.path.abspath(os.path.dirname(path))
    with self._lock:
        folder_handler = self._folder_handlers.get(folder_path)
        if folder_handler is None:
            _LOGGER.debug('Cannot stop watching path, because it is already not being watched. %s', folder_path)
            return
        folder_handler.remove_path_change_listener(path, callback)
        if not folder_handler.is_watching_paths():
            self._observer.unschedule(folder_handler.watch)
            del self._folder_handlers[folder_path]