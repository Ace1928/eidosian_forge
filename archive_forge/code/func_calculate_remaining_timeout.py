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
def calculate_remaining_timeout(*, timeout_s: Optional[float], start_time_s: float, curr_time_s: float) -> Optional[float]:
    """Get the timeout remaining given an overall timeout, start time, and curr time.

    If the timeout passed in was `None` or negative, will always return that timeout
    directly.

    If the timeout is >= 0, the returned remaining timeout always be >= 0.
    """
    if timeout_s is None or timeout_s < 0:
        return timeout_s
    time_since_start_s = curr_time_s - start_time_s
    return max(0, timeout_s - time_since_start_s)