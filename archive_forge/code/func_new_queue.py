from __future__ import annotations
import sys
from queue import Empty, Queue
from kombu.exceptions import reraise
from kombu.log import get_logger
from kombu.utils.objects import cached_property
from . import virtual
def new_queue(self, queue):
    if queue in self.queues:
        return
    self.queues[queue] = Queue()