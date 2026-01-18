import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def base_iterator(timeout=None):
    queue = collections.deque()
    ray.get(a.par_iter_init.remote(t))
    for _ in range(num_async):
        queue.append(a.par_iter_next_batch.remote(batch_ms))
    while True:
        try:
            batch = ray.get(queue.popleft(), timeout=timeout)
            queue.append(a.par_iter_next_batch.remote(batch_ms))
            for item in batch:
                yield item
            if timeout is not None:
                yield _NextValueNotReady()
        except TimeoutError:
            yield _NextValueNotReady()
        except StopIteration:
            break