import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import ray
from ray.util.annotations import PublicAPI
def put_nowait_batch(self, items):
    if self.maxsize > 0 and len(items) + self.qsize() > self.maxsize:
        raise Full(f'Cannot add {len(items)} items to queue of size {self.qsize()} and maxsize {self.maxsize}.')
    for item in items:
        self.queue.put_nowait(item)