from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
@property
def event_queue(self):
    """The event queue which is populated with file system events
        by emitters and from which events are dispatched by a dispatcher
        thread."""
    return self._event_queue