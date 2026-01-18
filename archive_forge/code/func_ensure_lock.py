from __future__ import annotations
import multiprocessing
import threading
import uuid
import weakref
from collections.abc import Hashable, MutableMapping
from typing import Any, ClassVar
from weakref import WeakValueDictionary
def ensure_lock(lock):
    """Ensure that the given object is a lock."""
    if lock is None or lock is False:
        return DummyLock()
    return lock