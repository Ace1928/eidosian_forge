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
def current_scheduler() -> 'Scheduler':
    """Gets the currently-running duet scheduler.

    This must be called from within a running async function, or else it will
    raise a RuntimeError.
    """
    return current_task().scheduler