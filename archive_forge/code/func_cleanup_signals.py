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
def cleanup_signals(self):
    if self._prev_signal:
        signal.signal(signal.SIGINT, self._prev_signal)