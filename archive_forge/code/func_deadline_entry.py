import enum
import functools
import heapq
import itertools
import signal
import threading
import time
from concurrent.futures import Future
from contextvars import ContextVar
from typing import (
import duet.futuretools as futuretools
@property
def deadline_entry(self) -> Optional['DeadlineEntry']:
    return self._deadlines[-1] if self._deadlines else None