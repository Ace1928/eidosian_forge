from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List
import ray
from ray.data.block import BlockAccessor, CallableClass
def locality_string(locality_hits: int, locality_misses) -> str:
    """Return a human-readable string for object locality stats."""
    if not locality_misses:
        return '[all objects local]'
    return f'[{locality_hits}/{locality_hits + locality_misses} objects local]'