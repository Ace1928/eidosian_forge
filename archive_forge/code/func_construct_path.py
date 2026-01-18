import abc
import functools
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import StartTraceback, find_free_port
from ray.exceptions import RayActorError
from ray.types import ObjectRef
def construct_path(path: Path, parent_path: Path) -> Path:
    """Constructs a path relative to a parent.

    Args:
        path: A relative or absolute path.
        parent_path: A relative path or absolute path.

    Returns: An absolute path.
    """
    if path.expanduser().is_absolute():
        return path.expanduser().resolve()
    else:
        return parent_path.joinpath(path).expanduser().resolve()