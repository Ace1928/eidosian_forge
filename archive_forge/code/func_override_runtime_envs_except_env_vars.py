import asyncio
import copy
import importlib
import inspect
import logging
import math
import os
import random
import string
import threading
import time
import traceback
from abc import ABC, abstractmethod
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
import requests
import ray
import ray.util.serialization_addons
from ray._private.resource_spec import HEAD_NODE_RESOURCE_NAME
from ray._private.utils import import_attr
from ray._private.worker import LOCAL_MODE, SCRIPT_MODE
from ray._raylet import MessagePackSerializer
from ray.actor import ActorHandle
from ray.exceptions import RayTaskError
from ray.serve._private.constants import HTTP_PROXY_TIMEOUT, SERVE_LOGGER_NAME
from ray.types import ObjectRef
from ray.util.serialization import StandaloneSerializationContext
def override_runtime_envs_except_env_vars(parent_env: Dict, child_env: Dict) -> Dict:
    """Creates a runtime_env dict by merging a parent and child environment.

    This method is not destructive. It leaves the parent and child envs
    the same.

    The merge is a shallow update where the child environment inherits the
    parent environment's settings. If the child environment specifies any
    env settings, those settings take precdence over the parent.
        - Note: env_vars are a special case. The child's env_vars are combined
            with the parent.

    Args:
        parent_env: The environment to inherit settings from.
        child_env: The environment with override settings.

    Returns: A new dictionary containing the merged runtime_env settings.

    Raises:
        TypeError: If a dictionary is not passed in for parent_env or child_env.
    """
    if not isinstance(parent_env, Dict):
        raise TypeError(f'Got unexpected type "{type(parent_env)}" for parent_env. parent_env must be a dictionary.')
    if not isinstance(child_env, Dict):
        raise TypeError(f'Got unexpected type "{type(child_env)}" for child_env. child_env must be a dictionary.')
    defaults = copy.deepcopy(parent_env)
    overrides = copy.deepcopy(child_env)
    default_env_vars = defaults.get('env_vars', {})
    override_env_vars = overrides.get('env_vars', {})
    defaults.update(overrides)
    default_env_vars.update(override_env_vars)
    defaults['env_vars'] = default_env_vars
    return defaults