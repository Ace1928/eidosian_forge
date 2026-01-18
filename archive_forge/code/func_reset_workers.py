from collections import defaultdict, deque
import logging
import platform
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type
import ray
from ray.actor import ActorClass, ActorHandle
def reset_workers(self, workers):
    """Notify that some workers may be removed."""
    for obj_ref, ev in self._tasks.copy().items():
        if ev not in workers:
            del self._tasks[obj_ref]
            del self._objects[obj_ref]
    for _ in range(len(self._fetching)):
        ev, obj_ref = self._fetching.popleft()
        if ev in workers:
            self._fetching.append((ev, obj_ref))