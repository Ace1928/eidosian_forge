import asyncio
from typing import Optional, Any, List, Dict
from collections.abc import Iterable
import ray
from ray.util.annotations import PublicAPI
def get_nowait(self):
    return self.queue.get_nowait()