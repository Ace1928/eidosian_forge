import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
def create_executable_class(executable_cls: Optional[Type]=None) -> Type:
    """Create the executable class to use as the Ray actors."""
    if not executable_cls:
        return RayTrainWorker
    elif issubclass(executable_cls, RayTrainWorker):
        return executable_cls
    else:

        class _WrappedExecutable(executable_cls, RayTrainWorker):

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
        return _WrappedExecutable