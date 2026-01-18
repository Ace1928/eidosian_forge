import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def par_iter_slice(self, step: int, start: int):
    """Iterates in increments of step starting from start."""
    assert self.local_it is not None, 'must call par_iter_init()'
    if self.next_ith_buffer is None:
        self.next_ith_buffer = collections.defaultdict(list)
    index_buffer = self.next_ith_buffer[start]
    if len(index_buffer) > 0:
        return index_buffer.pop(0)
    else:
        for j in range(step):
            try:
                val = next(self.local_it)
                self.next_ith_buffer[j].append(val)
            except StopIteration:
                pass
        if not self.next_ith_buffer[start]:
            raise StopIteration
    return self.next_ith_buffer[start].pop(0)