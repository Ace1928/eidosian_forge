import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def get_args(self) -> Tuple[Any]:
    """Return the tuple of arguments for this node."""
    return self._bound_args