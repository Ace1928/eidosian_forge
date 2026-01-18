import ray
from ray.dag.base import DAGNodeBase
from ray.dag.py_obj_scanner import _PyObjScanner
from ray.util.annotations import DeveloperAPI
from typing import (
import uuid
import asyncio
def get_kwargs(self) -> Dict[str, Any]:
    """Return the dict of keyword arguments for this node."""
    return self._bound_kwargs.copy()