import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def for_each(self, fn: Callable[[T], U], max_concurrency=1, resources=None) -> 'LocalIterator[U]':
    if max_concurrency == 1:

        def apply_foreach(it):
            for item in it:
                if isinstance(item, _NextValueNotReady):
                    yield item
                else:
                    while True:
                        with self._metrics_context():
                            result = fn(item)
                        yield result
                        if not isinstance(result, _NextValueNotReady):
                            break
    else:
        if resources is None:
            resources = {}

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
    if hasattr(fn, LocalIterator.ON_FETCH_START_HOOK_NAME):
        unwrapped = apply_foreach

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
        apply_foreach = add_wait_hooks
    return LocalIterator(self.base_iterator, self.shared_metrics, self.local_transforms + [apply_foreach], name=self.name + '.for_each()')