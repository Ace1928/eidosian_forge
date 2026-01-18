import os
import logging
import unicodedata
from threading import Thread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.events import (
from wandb_watchdog.observers.api import (
import AppKit
from FSEvents import (
from FSEvents import (
def read_events(self):
    """
        Returns a list or one or more events, or None if there are no more
        events to be read.
        """
    if not self.is_alive():
        return None
    return self._queue.get()