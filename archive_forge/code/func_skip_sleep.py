from __future__ import annotations
import sys
import threading
import time
import weakref
from typing import Any, Callable, Optional
from pymongo.lock import _create_lock
def skip_sleep(self) -> None:
    self._skip_sleep = True