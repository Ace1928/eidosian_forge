from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def get_path_watcher_class(watcher_type: str) -> PathWatcherType:
    """Return the PathWatcher class that corresponds to the given watcher_type
    string. Acceptable values are 'auto', 'watchdog', 'poll' and 'none'.
    """
    if watcher_type in {'watchdog', 'auto'} and _is_watchdog_available():
        from streamlit.watcher.event_based_path_watcher import EventBasedPathWatcher
        return EventBasedPathWatcher
    elif watcher_type == 'auto':
        return PollingPathWatcher
    elif watcher_type == 'poll':
        return PollingPathWatcher
    else:
        return NoOpPathWatcher