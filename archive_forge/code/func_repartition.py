import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def repartition(self, num_partitions: int, batch_ms: int=0) -> 'ParallelIterator[T]':
    """Returns a new ParallelIterator instance with num_partitions shards.

        The new iterator contains the same data in this instance except with
        num_partitions shards. The data is split in round-robin fashion for
        the new ParallelIterator.

        Args:
            num_partitions: The number of shards to use for the new
                ParallelIterator
            batch_ms: Batches items for batch_ms milliseconds
                on each shard before retrieving it.
                Increasing batch_ms increases latency but improves throughput.

        Returns:
            A ParallelIterator with num_partitions number of shards and the
            data of this ParallelIterator split round-robin among the new
            number of shards.

        Examples:
            >>> it = from_range(8, 2)
            >>> it = it.repartition(3)
            >>> list(it.get_shard(0))
            [0, 4, 3, 7]
            >>> list(it.get_shard(1))
            [1, 5]
            >>> list(it.get_shard(2))
            [2, 6]
        """
    all_actors = []
    for actor_set in self.actor_sets:
        actor_set.init_actors()
        all_actors.extend(actor_set.actors)

    def base_iterator(num_partitions, partition_index, timeout=None):
        futures = {}
        for a in all_actors:
            futures[a.par_iter_slice_batch.remote(step=num_partitions, start=partition_index, batch_ms=batch_ms)] = a
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
                    batch = ray.get(obj_ref)
                    futures[actor.par_iter_slice_batch.remote(step=num_partitions, start=partition_index, batch_ms=batch_ms)] = actor
                    for item in batch:
                        yield item
                except StopIteration:
                    pass
            if timeout is not None:
                yield _NextValueNotReady()

    def make_gen_i(i):
        return lambda: base_iterator(num_partitions, i)
    name = self.name + f'.repartition[num_partitions={num_partitions}]'
    generators = [make_gen_i(s) for s in range(num_partitions)]
    worker_cls = ray.remote(ParallelIteratorWorker)
    actors = [worker_cls.remote(g, repeat=False) for g in generators]
    return ParallelIterator([_ActorSet(actors, [])], name, parent_iterators=[self])