from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def slowadd(a, b, delay=0.1):
    time.sleep(delay)
    return a + b