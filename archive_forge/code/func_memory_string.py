from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List
import ray
from ray.data.block import BlockAccessor, CallableClass
def memory_string(num_bytes: int) -> str:
    """Return a human-readable memory string for the given amount of bytes."""
    if num_bytes >= 1024 * 1024 * 1024:
        mem = str(round(num_bytes / (1024 * 1024 * 1024), 2)) + ' GiB'
    else:
        mem = str(round(num_bytes / (1024 * 1024), 2)) + ' MiB'
    return mem