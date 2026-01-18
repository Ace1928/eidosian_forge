import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def apply_foreach(it):
    cur = []
    remote = ray.remote(fn).options(**resources)
    remote_fn = remote.remote
    for item in it:
        if isinstance(item, _NextValueNotReady):
            yield item
        else:
            if max_concurrency and len(cur) >= max_concurrency:
                finished, cur = ray.wait(cur)
                yield from ray.get(finished)
            cur.append(remote_fn(item))
    while cur:
        finished, cur = ray.wait(cur)
        yield from ray.get(finished)