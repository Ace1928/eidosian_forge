from __future__ import annotations
import queue
import threading
from pathlib import Path
from watchdog.utils import BaseThread, Protocol
from watchdog.utils.bricks import SkipRepeatsQueue
@property
def event_filter(self):
    """Collection of event types watched for the path"""
    return self._event_filter