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
def ray_address_to_api_server_url(address: Optional[str]) -> str:
    """Parse a Ray cluster address into API server URL.

    When an address is provided, it will be used to query GCS for
    API server address from GCS, so a Ray cluster must be running.

    When an address is not provided, it will first try to auto-detect
    a running Ray instance, or look for local GCS process.

    Args:
        address: Ray cluster bootstrap address or Ray Client address.
            Could also be `auto`.

    Returns:
        API server HTTP URL.
    """
    address = services.canonicalize_bootstrap_address_or_die(address)
    gcs_client = GcsClient(address=address, nums_reconnect_retry=0)
    ray.experimental.internal_kv._initialize_internal_kv(gcs_client)
    api_server_url = ray._private.utils.internal_kv_get_with_retry(gcs_client, ray_constants.DASHBOARD_ADDRESS, namespace=ray_constants.KV_NAMESPACE_DASHBOARD, num_retries=20)
    if api_server_url is None:
        raise ValueError("Couldn't obtain the API server address from GCS. It is likely that the GCS server is down. Check gcs_server.[out | err] to see if it is still alive.")
    api_server_url = f'http://{api_server_url.decode()}'
    return api_server_url