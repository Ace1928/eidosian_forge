import sys
import threading
import time
import traceback
from types import SimpleNamespace
def registerCallback(fn):
    """Register a callable to be invoked when there is an unhandled exception.
    The callback will be passed an object with attributes: [exc_type, exc_value, exc_traceback, thread]
    (see threading.excepthook).
    Multiple callbacks will be invoked in the order they were registered.
    """
    callbacks.append(fn)