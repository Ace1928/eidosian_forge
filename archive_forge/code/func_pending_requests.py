import collections
import datetime
import logging
import queue as queue_module
import threading
import time
from google.api_core import exceptions
@property
def pending_requests(self):
    """int: Returns an estimate of the number of queued requests."""
    return self._request_queue.qsize()