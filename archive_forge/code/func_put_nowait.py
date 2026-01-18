import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import ray
from ray.util.annotations import PublicAPI
def put_nowait(self, item):
    self.queue.put_nowait(item)