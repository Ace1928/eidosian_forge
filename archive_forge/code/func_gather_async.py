import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def gather_async(self, batch_ms=0, num_async=1) -> 'LocalIterator[T]':
    """Returns a local iterable for asynchronous iteration.

        New items will be fetched from the shards asynchronously as soon as
        the previous one is computed. Items arrive in non-deterministic order.

        Arguments:
            batch_ms: Batches items for batch_ms milliseconds
                on each shard before retrieving it.
                Increasing batch_ms increases latency but improves throughput.
                If this value is 0, then items are returned immediately.
            num_async: The max number of async requests in flight
                per actor. Increasing this improves the amount of pipeline
                parallelism in the iterator.

        Examples:
            >>> it = from_range(100, 1).gather_async()
            >>> next(it)
            ... 3
            >>> next(it)
            ... 0
            >>> next(it)
            ... 1
        """
    if num_async < 1:
        raise ValueError('queue depth must be positive')
    if batch_ms < 0:
        raise ValueError('batch time must be positive')
    local_iter = None

    def base_iterator(timeout=None):
        all_actors = []
        for actor_set in self.actor_sets:
            actor_set.init_actors()
            all_actors.extend(actor_set.actors)
        futures = {}
        for _ in range(num_async):
            for a in all_actors:
                futures[a.par_iter_next_batch.remote(batch_ms)] = a
        while futures:
            pending = list(futures)
            if timeout is None:
                ready, _ = ray.wait(pending, num_returns=len(pending), timeout=0)
                if not ready:
                    ready, _ = ray.wait(pending, num_returns=1)
            else:
                ready, _ = ray.wait(pending, num_returns=len(pending), timeout=timeout)
            for obj_ref in ready:
                actor = futures.pop(obj_ref)
                try:
                    local_iter.shared_metrics.get().current_actor = actor
                    batch = ray.get(obj_ref)
                    futures[actor.par_iter_next_batch.remote(batch_ms)] = actor
                    for item in batch:
                        yield item
                except StopIteration:
                    pass
            if timeout is not None:
                yield _NextValueNotReady()
    name = f'{self}.gather_async()'
    local_iter = LocalIterator(base_iterator, SharedMetrics(), name=name)
    return local_iter