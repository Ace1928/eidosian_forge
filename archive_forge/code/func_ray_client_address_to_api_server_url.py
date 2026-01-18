import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
def ray_client_address_to_api_server_url(address: str):
    """Convert a Ray Client address of a running Ray cluster to its API server URL.

    Args:
        address: The Ray Client address, e.g. "ray://my-cluster".

    Returns:
        str: The API server URL of the cluster, e.g. "http://<head-node-ip>:8265".
    """
    with ray.init(address=address) as client_context:
        dashboard_url = client_context.dashboard_url
    return f'http://{dashboard_url}'