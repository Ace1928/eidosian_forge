import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def local_shuffle(self, shuffle_buffer_size: int, seed: int=None) -> 'ParallelIterator[T]':
    """Remotely shuffle items of each shard independently

        Args:
            shuffle_buffer_size: The algorithm fills a buffer with
                shuffle_buffer_size elements and randomly samples elements from
                this buffer, replacing the selected elements with new elements.
                For perfect shuffling, this argument should be greater than or
                equal to the largest iterator size.
            seed: Seed to use for
                randomness. Default value is None.

        Returns:
            A ParallelIterator with a local shuffle applied on the base
            iterator

        Examples:
            >>> it = from_range(10, 1).local_shuffle(shuffle_buffer_size=2)
            >>> it = it.gather_sync()
            >>> next(it)
            0
            >>> next(it)
            2
            >>> next(it)
            3
            >>> next(it)
            1
        """
    return self._with_transform(lambda local_it: local_it.shuffle(shuffle_buffer_size, seed), '.local_shuffle(shuffle_buffer_size={}, seed={})'.format(shuffle_buffer_size, str(seed) if seed is not None else 'None'))