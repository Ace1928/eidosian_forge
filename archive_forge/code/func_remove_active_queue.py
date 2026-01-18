import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
@classmethod
def remove_active_queue(cls):
    """Ends recording on the currently active recording queue."""
    return cls._active_contexts.pop()