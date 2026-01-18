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
def any_ready(tasks: Set[Task]) -> futuretools.AwaitableFuture[None]:
    """Returns a Future that will fire when any of the given tasks is ready."""
    if not tasks or any((task.done for task in tasks)):
        return futuretools.completed_future(None)
    f = futuretools.AwaitableFuture[None]()
    for task in tasks:
        task.add_ready_callback(lambda _: f.try_set_result(None))
    return f