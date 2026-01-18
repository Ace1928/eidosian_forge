import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def gather_sync(self) -> 'LocalIterator[T]':
    """Returns a local iterable for synchronous iteration.

        New items will be fetched from the shards on-demand as the iterator
        is stepped through.

        This is the equivalent of batch_across_shards().flatten().

        Examples:
            >>> it = from_range(100, 1).gather_sync()
            >>> next(it)
            ... 0
            >>> next(it)
            ... 1
            >>> next(it)
            ... 2
        """
    it = self.batch_across_shards().flatten()
    it.name = f'{self}.gather_sync()'
    return it