import collections
import random
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Generic, Iterable, List, TypeVar
import ray
from ray.util.annotations import Deprecated
from ray.util.iter_metrics import MetricsContext, SharedMetrics
def num_shards(self) -> int:
    """Return the number of worker actors backing this iterator."""
    return sum((len(a.actors) for a in self.actor_sets))