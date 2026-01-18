import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def add_wait_hooks(it):
    it = unwrapped(it)
    new_item = True
    while True:
        if new_item:
            with self._metrics_context():
                fn._on_fetch_start()
            new_item = False
        item = next(it)
        if not isinstance(item, _NextValueNotReady):
            new_item = True
        yield item