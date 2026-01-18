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
def get_head_node_id() -> str:
    """Get the head node id.

    Iterate through all nodes in the ray cluster and return the node id of the first
    alive node with head node resource.
    """
    head_node_id = None
    for node in ray.nodes():
        if HEAD_NODE_RESOURCE_NAME in node['Resources'] and node['Alive']:
            head_node_id = node['NodeID']
            break
    assert head_node_id is not None, 'Cannot find alive head node.'
    return head_node_id