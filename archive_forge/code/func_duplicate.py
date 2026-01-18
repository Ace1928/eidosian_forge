import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def duplicate(self, n) -> List['LocalIterator[T]']:
    """Copy this iterator `n` times, duplicating the data.

        The child iterators will be prioritized by how much of the parent
        stream they have consumed. That is, we will not allow children to fall
        behind, since that can cause infinite memory buildup in this operator.

        Returns:
            List[LocalIterator[T]]: child iterators that each have a copy
                of the data of this iterator.
        """
    if n < 2:
        raise ValueError('Number of copies must be >= 2')
    queues = []
    for _ in range(n):
        queues.append(collections.deque())

    def fill_next(timeout):
        self.timeout = timeout
        item = next(self)
        for q in queues:
            q.append(item)

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
    iterators = []
    for i in range(n):
        iterators.append(LocalIterator(make_next(i), self.shared_metrics, [], name=self.name + f'.duplicate[{i}]'))
    return iterators