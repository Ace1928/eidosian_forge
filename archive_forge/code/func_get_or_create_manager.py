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
def get_or_create_manager(warn_on_creation: bool=True) -> 'ActorHandle':
    """Get or create the storage manager."""
    try:
        return ray.get_actor(common.STORAGE_ACTOR_NAME, namespace=common.MANAGEMENT_ACTOR_NAMESPACE)
    except ValueError:
        if warn_on_creation:
            logger.warning('Cannot access workflow serialization manager. It could be because the workflow manager exited unexpectedly. A new workflow manager is being created. ')
        handle = Manager.options(name=common.STORAGE_ACTOR_NAME, namespace=common.MANAGEMENT_ACTOR_NAMESPACE, lifetime='detached').remote()
        ray.get(handle.ping.remote())
        return handle