from __future__ import with_statement
import threading
from wandb_watchdog.utils import BaseThread
from wandb_watchdog.utils.compat import queue
from wandb_watchdog.utils.bricks import SkipRepeatsQueue
@property
def emitters(self):
    """Returns event emitter created by this observer."""
    return self._emitters