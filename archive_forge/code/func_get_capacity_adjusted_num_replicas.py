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
def get_capacity_adjusted_num_replicas(num_replicas: int, target_capacity: Optional[float]) -> int:
    """Return the `num_replicas` adjusted by the `target_capacity`.

    The output will only ever be 0 if `target_capacity` is 0 or `num_replicas` is
    0 (to support autoscaling deployments using scale-to-zero).

    Rather than using the default `round` behavior in Python, which rounds half to
    even, uses the `decimal` module to round half up (standard rounding behavior).
    """
    if target_capacity is None or target_capacity == 100:
        return num_replicas
    if target_capacity == 0 or num_replicas == 0:
        return 0
    adjusted_num_replicas = Decimal(num_replicas * target_capacity) / Decimal(100.0)
    rounded_adjusted_num_replicas = adjusted_num_replicas.to_integral_value(rounding=ROUND_HALF_UP)
    return max(1, int(rounded_adjusted_num_replicas))