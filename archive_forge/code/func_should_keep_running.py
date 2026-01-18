import os
import sys
import threading
from wandb_watchdog.utils import platform
from wandb_watchdog.utils.compat import Event
def should_keep_running(self):
    """Determines whether the thread should continue running."""
    return not self._stopped_event.is_set()