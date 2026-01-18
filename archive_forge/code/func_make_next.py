import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def make_next(i):

    def gen(timeout):
        while True:
            my_len = len(queues[i])
            max_len = max((len(q) for q in queues))
            if my_len < max_len:
                yield _NextValueNotReady()
            else:
                if len(queues[i]) == 0:
                    try:
                        fill_next(timeout)
                    except StopIteration:
                        return
                yield queues[i].popleft()
    return gen