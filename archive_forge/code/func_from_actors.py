import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
@Deprecated
def from_actors(actors: List['ray.actor.ActorHandle'], name=None) -> 'ParallelIterator[T]':
    """Create a parallel iterator from an existing set of actors.

    Each actor must subclass the ParallelIteratorWorker interface.

    Args:
        actors: List of actors that each implement
            ParallelIteratorWorker.
        name: Optional name to give the iterator.
    """
    if not name:
        name = f'from_actors[shards={len(actors)}]'
    return ParallelIterator([_ActorSet(actors, [])], name, parent_iterators=[])