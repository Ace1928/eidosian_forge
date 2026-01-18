import socket
import threading
import time
from collections import deque
from queue import Empty
from time import sleep
from weakref import WeakKeyDictionary
from kombu.utils.compat import detect_environment
from celery import states
from celery.exceptions import TimeoutError
from celery.utils.threads import THREAD_TIMEOUT_MAX
def register_drainer(name):
    """Decorator used to register a new result drainer type."""

    def _inner(cls):
        drainers[name] = cls
        return cls
    return _inner