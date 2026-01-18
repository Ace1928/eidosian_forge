import importlib
import inspect
import os
import warnings
from eventlet import patcher
from eventlet.support import greenlets as greenlet
from eventlet import timeout
def notify_close(fd):
    """
    A particular file descriptor has been explicitly closed. Register for any
    waiting listeners to be notified on the next run loop.
    """
    hub = get_hub()
    hub.notify_close(fd)