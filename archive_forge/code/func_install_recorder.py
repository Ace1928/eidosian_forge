import os
import json
import atexit
import abc
import enum
import time
import threading
from timeit import default_timer as timer
from contextlib import contextmanager, ExitStack
from collections import defaultdict
from numba.core import config
@contextmanager
def install_recorder(kind):
    """Install a RecordingListener temporarily to record all events.

    Once the context is closed, users can use ``RecordingListener.buffer``
    to access the recorded events.

    Returns
    -------
    res : RecordingListener

    Examples
    --------

    This is equivalent to:

    >>> with install_listener(kind, RecordingListener()) as res:
    >>>    ...
    """
    rl = RecordingListener()
    with install_listener(kind, rl):
        yield rl