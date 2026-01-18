import contextlib
from dataclasses import dataclass
import logging
import os
import ray
from ray import cloudpickle
from ray.types import ObjectRef
from ray.workflow import common, workflow_storage
from typing import Any, Dict, Generator, List, Optional, Tuple, TYPE_CHECKING
from collections import ChainMap
import io
@contextlib.contextmanager
def objectref_cache() -> Generator:
    """A reentrant caching context for object refs."""
    global _object_cache
    clear_cache = _object_cache is None
    if clear_cache:
        _object_cache = {}
    try:
        yield
    finally:
        if clear_cache:
            _object_cache = None