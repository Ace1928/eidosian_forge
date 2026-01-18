from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
def queue_events(self, timeout):
    """Override this method to populate the event queue with events
        per interval period.

        :param timeout:
            Timeout (in seconds) between successive attempts at
            reading events.
        :type timeout:
            ``float``
        """