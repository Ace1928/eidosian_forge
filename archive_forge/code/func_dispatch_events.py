from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def dispatch_events(self, event_queue, timeout):
    event, watch = event_queue.get(block=True, timeout=timeout)
    with self._lock:
        for handler in list(self._handlers.get(watch, [])):
            if handler in self._handlers.get(watch, []):
                handler.dispatch(event)
    event_queue.task_done()