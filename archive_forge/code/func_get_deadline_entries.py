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
def get_deadline_entries(self, deadline: float) -> Iterator[DeadlineEntry]:
    while self._deadlines and self._deadlines[0].deadline <= deadline:
        entry = heapq.heappop(self._deadlines)
        if entry.valid:
            yield entry