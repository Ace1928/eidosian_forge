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
def get_all_live_placement_group_names() -> List[str]:
    """Fetch and parse the Ray placement group table for live placement group names.

    Placement groups are filtered based on their `scheduling_state`; any placement
    group not in the "REMOVED" state is considered live.
    """
    placement_group_table = ray.util.placement_group_table()
    live_pg_names = []
    for entry in placement_group_table.values():
        pg_name = entry.get('name', '')
        if pg_name and entry.get('stats', {}).get('scheduling_state', 'UNKNOWN') != 'REMOVED':
            live_pg_names.append(pg_name)
    return live_pg_names