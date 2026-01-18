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
def from_iterators(generators: List[Iterable[T]], repeat: bool=False, name=None) -> 'ParallelIterator[T]':
    """Create a parallel iterator from a list of iterables.
    An iterable can be a conatiner (list, str, tuple, set, etc.),
    a generator, or a custom class that implements __iter__ or __getitem__.

    An actor will be created for each iterable.

    Examples:
        >>> # Create using a list of generators.
        >>> from_iterators([range(100), range(100)])

        >>> # Certain generators are not serializable.
        >>> from_iterators([(x for x in range(100))])
        ... TypeError: can't pickle generator objects

        >>> # So use lambda functions instead.
        >>> # Lambda functions are serializable.
        >>> from_iterators([lambda: (x for x in range(100))])

    Args:
        generators: A list of Python iterables or lambda
            functions that produce an iterable when called. We allow lambda
            functions since certain generators might not be serializable,
            but a lambda that returns it can be.
        repeat: Whether to cycle over the iterators forever.
        name: Optional name to give the iterator.
    """
    worker_cls = ray.remote(ParallelIteratorWorker)
    actors = [worker_cls.remote(g, repeat) for g in generators]
    if not name:
        name = 'from_iterators[shards={}{}]'.format(len(generators), ', repeat=True' if repeat else '')
    return from_actors(actors, name=name)